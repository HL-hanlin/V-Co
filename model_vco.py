# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov2_hf import RAE
from einops import rearrange
from projectors import ProjectionLayer
from util.model_util import (
    ConcatVisionRotaryEmbedding,
    get_2d_sincos_pos_embed,
    RMSNorm,
    VisionRotaryEmbeddingFast,
)

try:
    from transformers import AutoModel
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CoDINOv2GatedPatchEmbed(nn.Module):
    """
    Pixel patch embed + DINOv2 patch embed -> single fused token stream (B, N, D)
    using LayerNorm + gated residual fusion (stable, drop-in).

    Inputs:
      - x:  (B, 3, H, W) pixels for "jit" (patchified by conv)
      - zs: either
            (B, N, dinov2_dim) DINO patch tokens already extracted, OR
            (B, 3, H, W) raw pixels to run through DINOv2 (enable run_dinov2_in_forward=True)

    Output:
      - out: (B, N, embed_dim) fused tokens, ready to feed into your existing JiT unchanged
    """

    def __init__(
        self,
        img_size: int = 256,
        pixel_img_size: int = None,  # If None, defaults to img_size. Use when pixel and DINOv2 inputs differ in size.
        embed_dim: int = 768,
        dinov2_model_name: str = "facebook/dinov2-with-registers-base",
        freeze_dinov2: bool = True,
        run_dinov2_in_forward: bool = False,  # set True only if you want to pass raw pixels as zs
        patch_size: int = 16,
        in_chans: int = 3,
        pca_dim: int = 768,
        bias: bool = True,
        num_gate_hidden: int = 0,  # keep 0 for simplest; if >0, add a hidden layer in gate MLP
        gate_per_channel: bool = True,  # True: gate is (B,N,D), False: gate is (B,N,1)
        zero_init_dino_inject: bool = True,  # starts as "pixel-only"
        use_identity_proj_when_match: bool = True,  # if dinov2_dim==embed_dim, use Identity for proj
        hidden_size=None,
        use_mmdit: bool = False,
        jit_dino_proj: bool = False,
        jit_pixel_proj: bool = False,
        jit_refiner_layers: int = 2,
        num_heads: int = 12,
        use_conv2d_dino_proj: bool = False,
        use_dino_from_rae: bool = False,
        match_pixel_norm: float = 0.485,
        **kwargs,
    ):
        super().__init__()

        assert (
            DINOV2_AVAILABLE
        ), "transformers library is required for DINOv2. Install with: pip install transformers"

        # If pixel_img_size not specified, default to img_size
        if pixel_img_size is None:
            pixel_img_size = img_size

        self.img_size = (img_size, img_size)
        self.pixel_img_size = pixel_img_size
        self.patch_size_pix = patch_size
        self.patch_size_dino = 14  # DINOv2 base/large/giant typically uses 14
        self.num_patches = (img_size // self.patch_size_dino) ** 2
        self.embed_dim = embed_dim
        self.run_dinov2_in_forward = run_dinov2_in_forward
        self.use_mmdit = use_mmdit
        self.jit_dino_proj = jit_dino_proj
        self.jit_pixel_proj = jit_pixel_proj
        self.use_conv2d_dino_proj = use_conv2d_dino_proj
        self.use_dino_from_rae = use_dino_from_rae

        # Infer DINO output dim from model name (simple heuristic; override if you like)
        if "giant" in dinov2_model_name:
            self.dinov2_dim = 1536
        elif "large" in dinov2_model_name:
            self.dinov2_dim = 1024
        elif "small" in dinov2_model_name:
            self.dinov2_dim = 384
        else:
            # default/base
            self.dinov2_dim = 768

        self.dinov2 = RAE(match_pixel_norm=match_pixel_norm)

        # Freeze/unfreeze DINOv2 params
        for name, param in self.dinov2.named_parameters():
            # Always keep mask token frozen to avoid the "did not receive grad" warning
            if "embeddings.mask_token" in name:
                param.requires_grad = False
            else:
                param.requires_grad = not freeze_dinov2

        # Project DINO dim -> embed_dim
        if jit_dino_proj:
            # JIT-style refiner with self-attention and AdaLN
            # DINOv2 uses patch_size=14, so hw_seq_len = img_size // 14
            self.dino_proj = JITRefiner(
                input_dim=self.dinov2_dim,
                output_dim=embed_dim,
                num_layers=jit_refiner_layers,
                num_heads=num_heads,
                hw_seq_len=16,
            )
        else:
            if self.use_conv2d_dino_proj:
                self.dino_proj1 = nn.Conv2d(
                    self.dinov2_dim, pca_dim, kernel_size=1, stride=1, bias=False
                )
                self.dino_proj2 = nn.Conv2d(
                    pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias
                )
            else:
                # Original lightweight 2-layer MLP
                self.dino_proj = nn.Sequential(
                    nn.Linear(self.dinov2_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )

        # ---- Pixel patch embed ----
        # Patchify -> (B, pca_dim, H/ps, W/ps) then 1x1 -> embed_dim
        self.pix_proj1 = nn.Conv2d(
            in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.pix_proj2 = nn.Conv2d(
            pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias
        )

        # JIT-style pixel refiner (self-attention + AdaLN)
        if jit_pixel_proj:
            self.pix_refiner = JITRefiner(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_layers=jit_refiner_layers,
                num_heads=num_heads,
                hw_seq_len=16,
            )

        self.hidden_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,
        zs: torch.Tensor = None,
        t_emb: torch.Tensor = None,
        pixel_only: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:  (B, 3, H, W) pixels for jit branch (conv patchify)
            zs: either (B, N, dinov2_dim) DINO patch tokens, or (B, 3, H, W) pixels if run_dinov2_in_forward=True
            t_emb: (B, embed_dim) timestep embedding
            pixel_only: if True, only compute pixel tokens and return None for dino tokens (for inference)

        Returns:
            pix: (B, N, embed_dim) pixel tokens
            dino_tokens: (B, N, embed_dim) DINOv2 tokens (None if pixel_only)
        """

        # ---- 1) pixel tokens ----
        # (B, embed_dim, H/ps, W/ps) -> (B, N, embed_dim)
        pix = self.pix_proj2(self.pix_proj1(x)).flatten(2).transpose(1, 2)

        # Apply pixel projection/refiner
        if self.jit_pixel_proj:
            assert t_emb is not None, "t_emb required for jit_pixel_proj"
            pix = self.pix_refiner(pix, t_emb)

        if pixel_only:
            return pix, None

        # ---- 2) dino tokens ----
        if zs is None:
            raise ValueError(
                "zs must be provided (either DINO tokens or raw pixels if run_dinov2_in_forward=True)."
            )

        if self.jit_dino_proj:
            assert t_emb is not None, "t_emb required for jit_dino_proj"
            dino_tokens = self.dino_proj(zs, t_emb)
        elif self.use_conv2d_dino_proj:
            zs_2d = rearrange(zs, "b (h w) c -> b c h w", h=int(zs.shape[1] ** 0.5))
            dino_tokens = (
                self.dino_proj2(self.dino_proj1(zs_2d)).flatten(2).transpose(1, 2)
            )
        else:
            dino_tokens = self.dino_proj(zs)

        return pix, dino_tokens


class CoDINOv2PatchEmbed(nn.Module):
    """
    Simple Pixel patch embed + DINOv2 patch embed without gating.
    Returns separate pixel and dino token streams for co-denoising.

    This is a simpler alternative to CoDINOv2GatedPatchEmbed that doesn't use
    gated residual fusion, but still supports jit refiners.
    """

    def __init__(
        self,
        img_size: int = 256,
        pixel_img_size: int = None,  # If None, defaults to img_size
        embed_dim: int = 768,
        dinov2_model_name: str = "facebook/dinov2-with-registers-base",
        freeze_dinov2: bool = True,
        patch_size: int = 16,
        in_chans: int = 3,
        pca_dim: int = 768,
        pixel_embed_dim: int = 768,
        bias: bool = True,
        jit_dino_proj: bool = False,
        jit_pixel_proj: bool = False,
        jit_refiner_layers: int = 2,
        num_heads: int = 12,
        use_conv2d_dino_proj: bool = False,
        **kwargs,
    ):
        super().__init__()

        assert (
            DINOV2_AVAILABLE
        ), "transformers library is required for DINOv2. Install with: pip install transformers"

        # If pixel_img_size not specified, default to img_size
        if pixel_img_size is None:
            pixel_img_size = img_size

        self.img_size = (img_size, img_size)
        self.pixel_img_size = pixel_img_size
        self.patch_size_pix = patch_size
        self.patch_size_dino = 14  # DINOv2 base/large/giant typically uses 14
        self.num_patches = (img_size // self.patch_size_dino) ** 2
        self.embed_dim = embed_dim
        self.jit_dino_proj = jit_dino_proj
        self.jit_pixel_proj = jit_pixel_proj
        self.use_conv2d_dino_proj = use_conv2d_dino_proj

        # Infer DINO output dim from model name
        if "giant" in dinov2_model_name:
            self.dinov2_dim = 1536
        elif "large" in dinov2_model_name:
            self.dinov2_dim = 1024
        elif "small" in dinov2_model_name:
            self.dinov2_dim = 384
        else:
            self.dinov2_dim = 768

        # DINOv2 backbone
        self.dinov2 = AutoModel.from_pretrained(dinov2_model_name)

        # Freeze/unfreeze DINOv2 params
        for name, param in self.dinov2.named_parameters():
            if "embeddings.mask_token" in name:
                param.requires_grad = False
            else:
                param.requires_grad = not freeze_dinov2

        # Project DINO dim -> embed_dim
        if jit_dino_proj:
            # DINOv2 uses patch_size=14, so hw_seq_len = img_size // 14
            self.dino_proj = JITRefiner(
                input_dim=self.dinov2_dim,
                output_dim=embed_dim,
                num_layers=jit_refiner_layers,
                num_heads=num_heads,
                hw_seq_len=16,
            )
        else:
            self.dino_proj = nn.Sequential(
                nn.Linear(self.dinov2_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Pixel patch embed
        self.pix_proj1 = nn.Conv2d(
            in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.pix_proj2 = nn.Conv2d(
            pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias
        )

        # JIT-style pixel refiner
        if jit_pixel_proj:
            self.pix_refiner = JITRefiner(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_layers=jit_refiner_layers,
                num_heads=num_heads,
                hw_seq_len=16,
            )

    def forward(
        self,
        x: torch.Tensor,
        zs: torch.Tensor = None,
        t_emb: torch.Tensor = None,
        pixel_only: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) pixels
            zs: (B, N, dinov2_dim) DINO patch tokens
            t_emb: (B, embed_dim) timestep embedding for JITRefiner
            pixel_only: if True, only compute pixel tokens and return None for dino tokens (for inference)

        Returns:
            pix: (B, N, embed_dim) pixel tokens
            dino_tokens: (B, N, embed_dim) DINOv2 tokens (None if pixel_only)
        """
        # Pixel tokens
        pix = self.pix_proj2(self.pix_proj1(x)).flatten(2).transpose(1, 2)

        # Apply pixel refiner if enabled
        if self.jit_pixel_proj:
            assert t_emb is not None, "t_emb required for jit_pixel_proj"
            pix = self.pix_refiner(pix, t_emb)

        if pixel_only:
            return pix, None

        # DINO tokens
        if zs is None:
            raise ValueError("zs must be provided (DINO tokens).")

        if self.jit_dino_proj:
            assert t_emb is not None, "t_emb required for jit_dino_proj"
            dino_tokens = self.dino_proj(zs, t_emb)
        else:
            dino_tokens = self.dino_proj(zs)

        return pix, dino_tokens


class BottleneckPatchEmbed(nn.Module):
    """Image to Patch Embedding with optional Z-Image/JIT style refinement"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        pca_dim=768,
        embed_dim=768,
        bias=True,
        jit_pixel_proj=False,
        jit_refiner_layers=2,
        num_heads=12,
        **kwargs,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.jit_pixel_proj = jit_pixel_proj

        self.proj1 = nn.Conv2d(
            in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

        # JIT-style pixel refiner
        if jit_pixel_proj:
            self.pix_refiner = JITRefiner(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_layers=jit_refiner_layers,
                num_heads=num_heads,
                hw_seq_len=16,
            )

    def forward(self, x, zs=None, t_emb=None):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)

        # Apply pixel refiner if enabled
        if self.jit_pixel_proj:
            assert t_emb is not None, "t_emb required for jit_pixel_proj"
            x = self.pix_refiner(x, t_emb)

        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(
    query, key, value, dropout_p=0.0, attn_mask=None
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, attn_mask=attn_mask
    )


class REPAProjector(nn.Module):
    """
    REPA (Representation Alignment) projector.
    Uses the official REPA 3-layer MLP design with SiLU activation.
    Architecture: hidden_size -> projector_dim -> projector_dim -> target_dim
    """

    def __init__(
        self, hidden_size: int, target_dim: int = 768, projector_dim: int = 2048
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_dim = target_dim
        self.projector_dim = projector_dim

        # Official REPA MLP design: 3-layer MLP with SiLU
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, target_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_size) hidden states from pixel stream
        Returns:
            (B, N, target_dim) projected features for alignment with DINOv2
        """
        return self.proj(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        zs_dim=None,
        use_mmdit=False,
        jit_dino_head=False,
        jit_pixel_head=False,
        jit_refiner_layers=2,
        num_heads=12,
        hw_seq_len=16,
        **kwargs,
    ):
        super().__init__()

        self.zs_dim = zs_dim
        self.use_mmdit = use_mmdit
        self.jit_dino_head = jit_dino_head
        self.jit_pixel_head = jit_pixel_head
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Create RoPE for JiTBlock if jit heads are used
        if jit_pixel_head or jit_dino_head:
            half_head_dim = hidden_size // num_heads // 2
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.rope = None

        pixel_out_dim = patch_size * patch_size * out_channels

        if zs_dim is not None:
            self.pixels_adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            self.pixels_norm_final = RMSNorm(hidden_size)

            if jit_pixel_head:
                # JIT-style refiner + final projection for pixel output
                self.pixels_refiner = nn.ModuleList(
                    [
                        JiTBlock(
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=4.0,
                        )
                        for _ in range(jit_refiner_layers)
                    ]
                )
                self.pixels_linear = nn.Linear(hidden_size, pixel_out_dim, bias=True)
            else:
                # Simple linear projection (original design)
                self.pixels_linear = nn.Linear(hidden_size, pixel_out_dim, bias=True)

            self.dinov2_adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            self.dinov2_norm_final = RMSNorm(hidden_size)

            if jit_dino_head:
                # JIT-style refiner + final projection for DINOv2 output
                self.dinov2_refiner = nn.ModuleList(
                    [
                        JiTBlock(
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=4.0,
                        )
                        for _ in range(jit_refiner_layers)
                    ]
                )
                self.dinov2_linear = nn.Linear(hidden_size, zs_dim, bias=True)
            else:
                # Simple linear projection (original design)
                self.dinov2_linear = nn.Linear(hidden_size, zs_dim, bias=True)
        else:
            # No co_embed mode (pixel-only)
            self.norm_final = RMSNorm(hidden_size)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

            if jit_pixel_head:
                # JIT-style refiner + final projection for pixel output
                self.pixels_refiner = nn.ModuleList(
                    [
                        JiTBlock(
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=4.0,
                        )
                        for _ in range(jit_refiner_layers)
                    ]
                )
                self.linear = nn.Linear(hidden_size, pixel_out_dim, bias=True)
            else:
                self.linear = nn.Linear(hidden_size, pixel_out_dim, bias=True)

    @torch.compile
    def forward(self, x, c, x_dino=None):

        if self.zs_dim is not None:
            # Handle SharedJiT inference: x_dino=None means pixel-only mode
            if x_dino is None:
                # Pixel-only inference mode
                shift_pixels, scale_pixels = self.pixels_adaLN_modulation(c).chunk(
                    2, dim=1
                )
                x_pixels = modulate(
                    self.pixels_norm_final(x), shift_pixels, scale_pixels
                )
                if self.jit_pixel_head:
                    for block in self.pixels_refiner:
                        x_pixels = block(x_pixels, c, feat_rope=self.rope)
                    x_pixels = self.pixels_linear(x_pixels)
                else:
                    x_pixels = self.pixels_linear(x_pixels)
                # Return pixel output only (no concatenation with dinov2)
                return x_pixels
            elif self.use_mmdit:
                shift_pixels, scale_pixels = self.pixels_adaLN_modulation(c).chunk(
                    2, dim=1
                )
                x_pixels = modulate(
                    self.pixels_norm_final(x), shift_pixels, scale_pixels
                )
                if self.jit_pixel_head:
                    for block in self.pixels_refiner:
                        x_pixels = block(x_pixels, c, feat_rope=self.rope)
                    x_pixels = self.pixels_linear(x_pixels)
                else:
                    x_pixels = self.pixels_linear(x_pixels)

                shift_dinov2, scale_dinov2 = self.dinov2_adaLN_modulation(c).chunk(
                    2, dim=1
                )
                x_dinov2 = modulate(
                    self.dinov2_norm_final(x_dino), shift_dinov2, scale_dinov2
                )
                if self.jit_dino_head:
                    for block in self.dinov2_refiner:
                        x_dinov2 = block(x_dinov2, c, feat_rope=self.rope)
                    x_dinov2 = self.dinov2_linear(x_dinov2)
                else:
                    x_dinov2 = self.dinov2_linear(x_dinov2)

                return torch.cat([x_pixels, x_dinov2], dim=-1)
            else:
                # Single-stream mode: x contains pixel hidden states, x_dino contains dino hidden states
                shift_pixels, scale_pixels = self.pixels_adaLN_modulation(c).chunk(
                    2, dim=1
                )
                x_pixels = modulate(
                    self.pixels_norm_final(x), shift_pixels, scale_pixels
                )
                if self.jit_pixel_head:
                    for block in self.pixels_refiner:
                        x_pixels = block(x_pixels, c, feat_rope=self.rope)
                    x_pixels = self.pixels_linear(x_pixels)
                else:
                    x_pixels = self.pixels_linear(x_pixels)

                shift_dinov2, scale_dinov2 = self.dinov2_adaLN_modulation(c).chunk(
                    2, dim=1
                )
                x_dinov2 = modulate(
                    self.dinov2_norm_final(x_dino), shift_dinov2, scale_dinov2
                )
                if self.jit_dino_head:
                    for block in self.dinov2_refiner:
                        x_dinov2 = block(x_dinov2, c, feat_rope=self.rope)
                    x_dinov2 = self.dinov2_linear(x_dinov2)
                else:
                    x_dinov2 = self.dinov2_linear(x_dinov2)

                return torch.cat([x_pixels, x_dinov2], dim=-1)
        else:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            if self.jit_pixel_head:
                for block in self.pixels_refiner:
                    x = block(x, c, feat_rope=self.rope)
            x = self.linear(x)
            return x


class JiTBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class JITRefiner(nn.Module):
    """
    JIT-style refiner consisting of multiple JiTBlocks.
    Used as input projection for pixels or DINOv2 features with timestep conditioning.

    This uses JiTBlock architecture
    with 6-parameter AdaLN modulation instead of Z-Image's 4-parameter pattern.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        hw_seq_len: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size=output_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        half_head_dim = output_dim // num_heads // 2

        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim) input tokens
            t_emb: (B, output_dim) timestep embedding

        Returns:
            (B, N, output_dim) refined tokens
        """

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, t_emb, feat_rope=self.rope)

        return x


class MMDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        separate_qkv=False,
    ):
        super().__init__()
        self.separate_qkv = separate_qkv
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1_pixel = RMSNorm(hidden_size, eps=1e-6)
        self.norm1_dino = RMSNorm(hidden_size, eps=1e-6)

        if separate_qkv:
            # Separate QKV projections for each modality
            self.qkv_pixel = nn.Linear(hidden_size, hidden_size * 3, bias=True)
            self.qkv_dino = nn.Linear(hidden_size, hidden_size * 3, bias=True)

            # Separate Q/K norms for each modality
            self.q_norm_pixel = RMSNorm(self.head_dim)
            self.k_norm_pixel = RMSNorm(self.head_dim)
            self.q_norm_dino = RMSNorm(self.head_dim)
            self.k_norm_dino = RMSNorm(self.head_dim)

            # Separate output projections
            self.proj_pixel = nn.Linear(hidden_size, hidden_size)
            self.proj_dino = nn.Linear(hidden_size, hidden_size)

            self.attn_drop = nn.Dropout(attn_drop)
        else:
            # Shared attention (original behavior)
            self.attn = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )

        self.norm2_pixel = RMSNorm(hidden_size, eps=1e-6)
        self.norm2_dino = RMSNorm(hidden_size, eps=1e-6)

        self.mlp_pixel = SwiGLUFFN(
            hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop
        )
        self.mlp_dino = SwiGLUFFN(
            hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop
        )

        self.adaLN_modulation_pixel = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_dino = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(
        self,
        x_pixel,
        x_dino,
        c,
        feat_rope,
        block_dino_to_pixel=None,
        symmetric_attn_mask=False,
    ):

        c_dino = c

        B = x_pixel.shape[0]
        N_p = x_pixel.shape[1]
        N_d = x_dino.shape[1]
        C = x_pixel.shape[2]

        (
            shift_msa_pix,
            scale_msa_pix,
            gate_msa_pix,
            shift_mlp_pix,
            scale_mlp_pix,
            gate_mlp_pix,
        ) = self.adaLN_modulation_pixel(c).chunk(6, dim=-1)
        mod_x_pixel = modulate(self.norm1_pixel(x_pixel), shift_msa_pix, scale_msa_pix)

        (
            shift_msa_dino,
            scale_msa_dino,
            gate_msa_dino,
            shift_mlp_dino,
            scale_mlp_dino,
            gate_mlp_dino,
        ) = self.adaLN_modulation_dino(c_dino).chunk(6, dim=-1)
        mod_x_dino = modulate(self.norm1_dino(x_dino), shift_msa_dino, scale_msa_dino)

        # Construct attention mask for blocking cross-modal attention (attn_mask mode)
        attn_mask = None
        if block_dino_to_pixel is not None:
            N_total = N_p + N_d
            attn_mask = torch.zeros(
                B,
                1,
                N_total,
                N_total,
                device=x_pixel.device,
                dtype=x_pixel.dtype,
            )
            # Per-sample mask values: -inf for masked samples, 0 for unmasked
            mask_vals = torch.where(
                block_dino_to_pixel,
                torch.tensor(float("-inf"), device=x_pixel.device, dtype=x_pixel.dtype),
                torch.tensor(0.0, device=x_pixel.device, dtype=x_pixel.dtype),
            )  # (B,)
            expanded = mask_vals.view(B, 1, 1, 1)
            # Block pixel_query → DINO_key (top-right quadrant) — always applied
            attn_mask[:, :, :N_p, N_p:] = expanded.expand(B, 1, N_p, N_d)
            if symmetric_attn_mask:
                # Also block DINO_query → pixel_key (bottom-left quadrant)
                attn_mask[:, :, N_p:, :N_p] = expanded.expand(B, 1, N_d, N_p)

        if self.separate_qkv:
            # Separate QKV projections for each modality
            qkv_pixel = (
                self.qkv_pixel(mod_x_pixel)
                .reshape(B, N_p, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q_p, k_p, v_p = qkv_pixel[0], qkv_pixel[1], qkv_pixel[2]

            qkv_dino = (
                self.qkv_dino(mod_x_dino)
                .reshape(B, N_d, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q_d, k_d, v_d = qkv_dino[0], qkv_dino[1], qkv_dino[2]

            # Separate Q/K norms for each modality
            q_p = self.q_norm_pixel(q_p)
            k_p = self.k_norm_pixel(k_p)
            q_p = feat_rope(q_p)
            k_p = feat_rope(k_p)

            q_d = self.q_norm_dino(q_d)
            k_d = self.k_norm_dino(k_d)
            q_d = feat_rope(q_d)
            k_d = feat_rope(k_d)

            # Joint attention
            q = torch.cat([q_p, q_d], dim=2)
            k = torch.cat([k_p, k_d], dim=2)
            v = torch.cat([v_p, v_d], dim=2)

            attn_out = scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, N_p + N_d, C)

            # Separate output projections
            attn_pixel_out, attn_dino_out = torch.split(attn_out, [N_p, N_d], dim=1)
            attn_pixel = self.proj_pixel(attn_pixel_out)
            attn_dino = self.proj_dino(attn_dino_out)
        else:
            # Shared QKV (original behavior)
            mod_x_concat = torch.cat([mod_x_pixel, mod_x_dino], dim=1)
            N_cat = N_p + N_d
            qkv = (
                self.attn.qkv(mod_x_concat)
                .reshape(B, N_cat, 3, self.attn.num_heads, C // self.attn.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            (q_p, q_d) = torch.split(q, [N_p, N_d], dim=2)
            (k_p, k_d) = torch.split(k, [N_p, N_d], dim=2)
            (v_p, v_d) = torch.split(v, [N_p, N_d], dim=2)

            q_p = self.attn.q_norm(q_p)
            k_p = self.attn.k_norm(k_p)
            q_p = feat_rope(q_p)
            k_p = feat_rope(k_p)

            q_d = self.attn.q_norm(q_d)
            k_d = self.attn.k_norm(k_d)
            q_d = feat_rope(q_d)
            k_d = feat_rope(k_d)

            q = torch.cat([q_p, q_d], dim=2)
            k = torch.cat([k_p, k_d], dim=2)
            v = torch.cat([v_p, v_d], dim=2)

            attn_out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, N_p + N_d, C)
            attn_out = self.attn.proj(attn_out)

            attn_pixel, attn_dino = torch.split(attn_out, [N_p, N_d], dim=1)

        x_pixel = x_pixel + gate_msa_pix.unsqueeze(1) * attn_pixel
        x_pixel = x_pixel + gate_mlp_pix.unsqueeze(1) * self.mlp_pixel(
            modulate(self.norm2_pixel(x_pixel), shift_mlp_pix, scale_mlp_pix)
        )

        x_dino = x_dino + gate_msa_dino.unsqueeze(1) * attn_dino
        x_dino = x_dino + gate_mlp_dino.unsqueeze(1) * self.mlp_dino(
            modulate(self.norm2_dino(x_dino), shift_mlp_dino, scale_mlp_dino)
        )

        return x_pixel, x_dino


class JiT(nn.Module):
    """
    Just image Transformer.
    """

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        use_dinov2=False,
        dinov2_model_name="facebook/dinov2-with-registers-base",
        freeze_dinov2=True,
        use_co_embed=False,
        use_gated_co_embed=False,
        co_fuse="concat",
        use_mmdit: bool = False,
        separate_qkv: bool = False,
        jit_dino_proj: bool = False,
        jit_pixel_proj: bool = False,
        jit_dino_head: bool = False,
        jit_pixel_head: bool = False,
        jit_refiner_layers: int = 2,
        use_shared_jit: bool = False,
        use_channel_concat: bool = False,
        use_direct_addition: bool = False,
        use_conv2d_dino_proj: bool = False,
        use_dino_from_rae: bool = False,
        match_pixel_norm: float = 0.485,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.use_dinov2 = use_dinov2
        self.use_mmdit = use_mmdit

        self.use_co_embed = use_co_embed
        self.use_gated_co_embed = use_gated_co_embed
        self.separate_qkv = separate_qkv
        self.jit_dino_proj = jit_dino_proj
        self.jit_pixel_proj = jit_pixel_proj
        self.jit_dino_head = jit_dino_head
        self.jit_pixel_head = jit_pixel_head
        self.jit_refiner_layers = jit_refiner_layers
        self.use_shared_jit = use_shared_jit
        self.use_channel_concat = use_channel_concat
        self.use_direct_addition = use_direct_addition
        self.use_conv2d_dino_proj = use_conv2d_dino_proj
        self.use_dino_from_rae = use_dino_from_rae

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # zs_dim = hidden_size

        # linear embed
        if use_dinov2 and use_co_embed:
            # DINOv2: 224x224 input with patch size 14 -> 16x16 = 256 patches
            # Output: 256x256 with patch size 16 -> 16x16 grid
            dinov2_input_size = 224
            if self.use_gated_co_embed:
                self.x_embedder = CoDINOv2GatedPatchEmbed(
                    img_size=dinov2_input_size,
                    embed_dim=hidden_size,  # hidden_size,
                    dinov2_model_name=dinov2_model_name,
                    freeze_dinov2=freeze_dinov2,
                    patch_size=16,
                    in_chans=3,
                    pca_dim=bottleneck_dim,
                    bias=True,
                    hidden_size=hidden_size,
                    use_mmdit=self.use_mmdit,
                    jit_dino_proj=self.jit_dino_proj,
                    jit_pixel_proj=self.jit_pixel_proj,
                    jit_refiner_layers=self.jit_refiner_layers,
                    num_heads=num_heads,
                    use_conv2d_dino_proj=self.use_conv2d_dino_proj,
                    use_dino_from_rae=self.use_dino_from_rae,
                    match_pixel_norm=match_pixel_norm,
                )
            else:
                self.x_embedder = CoDINOv2PatchEmbed(
                    img_size=dinov2_input_size,
                    embed_dim=hidden_size,
                    dinov2_model_name=dinov2_model_name,
                    freeze_dinov2=freeze_dinov2,
                    ### pixels
                    patch_size=16,
                    in_chans=3,
                    pca_dim=bottleneck_dim,
                    pixel_embed_dim=768,
                    bias=True,
                    jit_dino_proj=self.jit_dino_proj,
                    jit_pixel_proj=self.jit_pixel_proj,
                    jit_refiner_layers=self.jit_refiner_layers,
                    num_heads=num_heads,
                    use_conv2d_dino_proj=self.use_conv2d_dino_proj,
                )
            self.encoder_patch_size = 14  # DINOv2 patch size
            self.decoder_patch_size = 16  # For decoding to 256x256
            num_patches = 256  # 16x16
            hw_seq_len = 16  # 16x16 grid
        else:
            self.x_embedder = BottleneckPatchEmbed(
                input_size,
                patch_size,
                in_channels,
                bottleneck_dim,
                hidden_size,
                bias=True,
                jit_pixel_proj=self.jit_pixel_proj,
                jit_refiner_layers=self.jit_refiner_layers,
                num_heads=num_heads,
            )
            self.decoder_patch_size = 16  # For decoding to 256x256

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True
            )
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)

        # rope - based on grid size
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        # RoPE for concatenated pixel+dino sequence in single-stream mode (2x sequence length)
        if not self.use_mmdit and self.use_co_embed:  # TODO
            # For single-stream with co-denoising, we have pixel tokens + dino tokens
            # Both represent the same 16x16 grid, so we use the same RoPE for both halves
            self.feat_rope_concat = ConcatVisionRotaryEmbedding(
                self.feat_rope, num_cls_token=0
            )
            self.feat_rope_concat_incontext = ConcatVisionRotaryEmbedding(
                self.feat_rope, num_cls_token=self.in_context_len
            )

        # transformer
        if self.use_shared_jit and self.use_co_embed:
            # SharedJiT architecture:
            # - K separate pre-blocks for pixel and DINOv2
            # - M shared middle blocks (both streams processed independently through same blocks)
            # - K separate post-blocks for pixel and DINOv2
            # K = jit_refiner_layers, M = depth - 2*K
            if self.use_channel_concat or self.use_direct_addition:
                # Channel concat / direct addition: no pre-blocks, only shared + post-blocks
                K = jit_refiner_layers
                M = depth - K
                assert M > 0, f"depth ({depth}) must be > jit_refiner_layers ({K})"
            else:
                K = jit_refiner_layers
                M = depth - 2 * K
                assert (
                    M > 0
                ), f"depth ({depth}) must be > 2 * jit_refiner_layers ({2 * K})"

            # Pre-processing blocks (separate for each stream) — only used when NOT channel_concat
            self.pixel_pre_blocks = nn.ModuleList(
                [
                    JiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(
                        K
                        if not (self.use_channel_concat or self.use_direct_addition)
                        else 0
                    )
                ]
            )
            self.dino_pre_blocks = nn.ModuleList(
                [
                    JiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(
                        K
                        if not (self.use_channel_concat or self.use_direct_addition)
                        else 0
                    )
                ]
            )

            # Shared middle blocks (both streams go through same blocks)
            self.shared_blocks = nn.ModuleList(
                [
                    JiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=(attn_drop if (M // 4 * 3 > i >= M // 4) else 0.0),
                        proj_drop=(proj_drop if (M // 4 * 3 > i >= M // 4) else 0.0),
                    )
                    for i in range(M)
                ]
            )

            # Post-processing blocks (separate for each stream)
            self.pixel_post_blocks = nn.ModuleList(
                [
                    JiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(K)
                ]
            )
            self.dino_post_blocks = nn.ModuleList(
                [
                    JiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(K)
                ]
            )

            # Total depth for SharedJiT = 2*K + M
            self.shared_jit_total_depth = 2 * K + M

            # Channel concat projection layers (2*hidden_size <-> hidden_size)
            if self.use_channel_concat:
                self.channel_concat_proj_in = nn.Linear(
                    2 * hidden_size, hidden_size, bias=True
                )
                self.channel_concat_proj_out = nn.Linear(
                    hidden_size, 2 * hidden_size, bias=True
                )

            # Set blocks to None to avoid confusion (not used in SharedJiT mode)
            self.blocks = None
        elif self.use_mmdit:
            self.blocks = nn.ModuleList(
                [
                    MMDiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=(
                            attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
                        ),
                        proj_drop=(
                            proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
                        ),
                        separate_qkv=separate_qkv,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    JiTBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=(
                            attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
                        ),
                        proj_drop=(
                            proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
                        ),
                    )
                    for i in range(depth)
                ]
            )

        # linear predict - always use decoder_patch_size for output
        if self.use_co_embed:
            # Get dinov2_dim from the embedder (automatically set based on model name)
            zs_dim = self.x_embedder.dinov2_dim
            self.final_layer = FinalLayer(
                hidden_size,
                self.decoder_patch_size,
                self.out_channels,
                zs_dim=zs_dim,
                use_mmdit=self.use_mmdit,
                jit_dino_head=self.jit_dino_head,
                jit_pixel_head=self.jit_pixel_head,
                jit_refiner_layers=self.jit_refiner_layers,
                num_heads=num_heads,
                hw_seq_len=hw_seq_len,
            )
        else:
            self.final_layer = FinalLayer(
                hidden_size,
                self.decoder_patch_size,
                self.out_channels,
                jit_pixel_head=self.jit_pixel_head,
                jit_refiner_layers=self.jit_refiner_layers,
                num_heads=num_heads,
                hw_seq_len=hw_seq_len,
            )

        # REPA projectors will be initialized separately via init_repa_projectors()
        self.repa_projectors = None
        self.repa_layers = None

        self.initialize_weights()

    def init_repa_projectors(self, repa_layers: list, target_dim: int = 768):
        """
        Initialize REPA projectors for the specified layers.
        Args:
            repa_layers: List of layer indices to extract hidden states from
            target_dim: Target dimension for projection (DINOv2 feature dim, default 768)
        """
        self.repa_layers = repa_layers
        self.repa_projectors = nn.ModuleDict(
            {
                str(layer_idx): REPAProjector(self.hidden_size, target_dim)
                for layer_idx in repa_layers
            }
        )
        # Move to same device as model
        device = next(self.parameters()).device
        self.repa_projectors.to(device)

    def initialize_weights(self):

        # # Initialize transformer layers:
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # self.apply(_basic_init)

        for name, module in self.named_modules():

            if isinstance(module, nn.Linear):
                # Skip everything under self.x_embedder.dinov2
                if "x_embedder.dinov2" in name:
                    print("skip", name)
                    continue

                print(f"Initializing module: {name}")
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # CoDINOv2*PatchEmbed uses pix_proj1/pix_proj2, BottleneckPatchEmbed uses proj1/proj2
        if self.use_co_embed and self.use_dinov2:
            # CoDINOv2GatedPatchEmbed or CoDINOv2PatchEmbed
            w1 = self.x_embedder.pix_proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.pix_proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.pix_proj2.bias, 0)
        else:
            # BottleneckPatchEmbed
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        if self.use_shared_jit and self.use_co_embed:
            # SharedJiT: initialize separate pre/post blocks and shared blocks
            for block in self.pixel_pre_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.dino_pre_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.shared_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.pixel_post_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.dino_post_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                if isinstance(block, MMDiTBlock):
                    nn.init.constant_(block.adaLN_modulation_pixel[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation_pixel[-1].bias, 0)
                    nn.init.constant_(block.adaLN_modulation_dino[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation_dino[-1].bias, 0)
                    # Initialize separate QKV projections if enabled
                    if block.separate_qkv:
                        nn.init.xavier_uniform_(block.qkv_pixel.weight)
                        nn.init.zeros_(block.qkv_pixel.bias)
                        nn.init.xavier_uniform_(block.qkv_dino.weight)
                        nn.init.zeros_(block.qkv_dino.bias)
                        nn.init.xavier_uniform_(block.proj_pixel.weight)
                        nn.init.zeros_(block.proj_pixel.bias)
                        nn.init.xavier_uniform_(block.proj_dino.weight)
                        nn.init.zeros_(block.proj_dino.bias)
                else:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.use_co_embed:
            # pass
            nn.init.constant_(self.final_layer.pixels_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.pixels_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(
                self.final_layer.dinov2_adaLN_modulation[-1].weight, 0
            )
            nn.init.constant_(self.final_layer.dinov2_adaLN_modulation[-1].bias, 0)

            # Zero-out pixel output layers
            if self.jit_pixel_head:
                nn.init.constant_(self.final_layer.pixels_linear.weight, 0)
                nn.init.constant_(self.final_layer.pixels_linear.bias, 0)
                for refiner_block in self.final_layer.pixels_refiner:
                    nn.init.constant_(refiner_block.adaLN_modulation[1].weight, 0)
                    nn.init.constant_(refiner_block.adaLN_modulation[1].bias, 0)
            else:
                nn.init.constant_(self.final_layer.pixels_linear.weight, 0)
                nn.init.constant_(self.final_layer.pixels_linear.bias, 0)

            # Zero-out dino output layers
            if self.jit_dino_head:
                nn.init.constant_(self.final_layer.dinov2_linear.weight, 0)
                nn.init.constant_(self.final_layer.dinov2_linear.bias, 0)
                for refiner_block in self.final_layer.dinov2_refiner:
                    nn.init.constant_(refiner_block.adaLN_modulation[1].weight, 0)
                    nn.init.constant_(refiner_block.adaLN_modulation[1].bias, 0)
            else:
                nn.init.constant_(self.final_layer.dinov2_linear.weight, 0)
                nn.init.constant_(self.final_layer.dinov2_linear.bias, 0)

        else:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

            # Zero-out pixel output layers
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """

        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        if self.use_co_embed:
            c_pixels = p * p * c
            x_pixels = x[:, :, :c_pixels]
            x_dinov2 = x[:, :, c_pixels:]

            x_pixels = x_pixels.reshape(shape=(x_pixels.shape[0], h, w, p, p, c))
            x_pixels = torch.einsum("nhwpqc->nchpwq", x_pixels)
            imgs = x_pixels.reshape(shape=(x_pixels.shape[0], c, h * p, h * p))
            return imgs, x_dinov2
        else:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum("nhwpqc->nchpwq", x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
            return imgs

    def forward(
        self,
        x,
        t,
        y,
        return_hidden_states=False,
        hidden_state_layers=None,
        zs=None,
        return_repa_hidden_states=False,
        inference_pixel_only=False,
        block_dino_to_pixel=None,
        symmetric_attn_mask=False,
    ):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        return_repa_hidden_states: if True, return hidden states for REPA alignment
        inference_pixel_only: if True, only process pixel stream (for SharedJiT inference)
        """

        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        hidden_states = {} if return_hidden_states else None

        # Collect hidden states for REPA if enabled
        repa_hidden_states = (
            {} if (return_repa_hidden_states and self.repa_layers is not None) else None
        )

        # Pass t_emb to x_embedder if JiT refiners are used
        embedder_t_emb = t_emb if (self.jit_pixel_proj or self.jit_dino_proj) else None

        if self.use_shared_jit and self.use_co_embed:
            # SharedJiT architecture: separate pre/post blocks, shared middle blocks

            if inference_pixel_only:
                # Inference mode: only process pixel stream (discard DINOv2 blocks)
                x_pixel, _ = self.x_embedder(
                    x, zs=None, t_emb=embedder_t_emb, pixel_only=True
                )
                x_pixel = x_pixel + self.pos_embed

                rope = self.feat_rope
                layer_idx = 0  # Global layer index for in-context tokens
                in_context_added = False

                # Pre-processing: K pixel-only blocks
                for block in self.pixel_pre_blocks:
                    if (
                        self.in_context_len > 0
                        and layer_idx >= self.in_context_start
                        and not in_context_added
                    ):
                        in_context_tokens = y_emb.unsqueeze(1).repeat(
                            1, self.in_context_len, 1
                        )
                        in_context_tokens += self.in_context_posemb
                        x_pixel = torch.cat([in_context_tokens, x_pixel], dim=1)
                        in_context_added = True
                    if in_context_added:
                        rope = self.feat_rope_incontext
                    x_pixel = block(x_pixel, c, rope)
                    layer_idx += 1

                # Shared middle: M blocks (pixel only)
                for block in self.shared_blocks:
                    if (
                        self.in_context_len > 0
                        and layer_idx >= self.in_context_start
                        and not in_context_added
                    ):
                        in_context_tokens = y_emb.unsqueeze(1).repeat(
                            1, self.in_context_len, 1
                        )
                        in_context_tokens += self.in_context_posemb
                        x_pixel = torch.cat([in_context_tokens, x_pixel], dim=1)
                        in_context_added = True
                    if in_context_added:
                        rope = self.feat_rope_incontext
                    x_pixel = block(x_pixel, c, rope)
                    layer_idx += 1

                # Post-processing: K pixel-only blocks
                for block in self.pixel_post_blocks:
                    if (
                        self.in_context_len > 0
                        and layer_idx >= self.in_context_start
                        and not in_context_added
                    ):
                        in_context_tokens = y_emb.unsqueeze(1).repeat(
                            1, self.in_context_len, 1
                        )
                        in_context_tokens += self.in_context_posemb
                        x_pixel = torch.cat([in_context_tokens, x_pixel], dim=1)
                        in_context_added = True
                    if in_context_added:
                        rope = self.feat_rope_incontext
                    x_pixel = block(x_pixel, c, rope)
                    layer_idx += 1

                # Remove in-context tokens before final layer
                if self.in_context_len > 0 and in_context_added:
                    x_pixel = x_pixel[:, self.in_context_len :]

                # Final layer (pixel only, no DINOv2)
                x = self.final_layer(x_pixel, c, x_dino=None)
            else:
                # Training mode: process both streams
                x_pixel, x_dino = self.x_embedder(x, zs=zs, t_emb=embedder_t_emb)
                x_pixel = x_pixel + self.pos_embed
                x_dino = x_dino + self.pos_embed

                rope = self.feat_rope
                layer_idx = 0  # Global layer index for in-context tokens
                in_context_added = False

                # Pre-processing: K separate blocks for each stream
                for i, (pixel_block, dino_block) in enumerate(
                    zip(self.pixel_pre_blocks, self.dino_pre_blocks)
                ):
                    if (
                        self.in_context_len > 0
                        and layer_idx >= self.in_context_start
                        and not in_context_added
                    ):
                        in_context_tokens = y_emb.unsqueeze(1).repeat(
                            1, self.in_context_len, 1
                        )
                        in_context_tokens += self.in_context_posemb
                        x_pixel = torch.cat([in_context_tokens, x_pixel], dim=1)
                        x_dino = torch.cat([in_context_tokens.clone(), x_dino], dim=1)
                        in_context_added = True
                    if in_context_added:
                        rope = self.feat_rope_incontext
                    x_pixel = pixel_block(x_pixel, c, rope)
                    x_dino = dino_block(x_dino, c, rope)
                    layer_idx += 1

                # Shared middle: M blocks (both streams go through same blocks, but processed independently)
                if self.use_channel_concat:
                    # Channel-concatenate pixel and dino along feature dim
                    x_concat = torch.cat(
                        [x_pixel, x_dino], dim=2
                    )  # (B, N, 2*hidden_size)
                    x_concat = self.channel_concat_proj_in(
                        x_concat
                    )  # (B, N, hidden_size)
                    for i, block in enumerate(self.shared_blocks):
                        x_concat = block(x_concat, c, rope)

                        # Collect hidden states for REPA
                        if repa_hidden_states is not None and i in self.repa_layers:
                            if in_context_added:
                                repa_hidden_states[i] = x_concat[
                                    :, self.in_context_len :
                                ]
                            else:
                                repa_hidden_states[i] = x_concat
                        layer_idx += 1
                    # Project back and split along feature dim
                    x_concat = self.channel_concat_proj_out(
                        x_concat
                    )  # (B, N, 2*hidden_size)
                    H = x_pixel.shape[2]
                    x_pixel = x_concat[:, :, :H]
                    x_dino = x_concat[:, :, H:]
                elif self.use_direct_addition:
                    # Element-wise add pixel and dino features
                    x_added = x_pixel + x_dino  # (B, N, hidden_size)
                    for i, block in enumerate(self.shared_blocks):
                        x_added = block(x_added, c, rope)

                        # Collect hidden states for REPA
                        if repa_hidden_states is not None and i in self.repa_layers:
                            if in_context_added:
                                repa_hidden_states[i] = x_added[
                                    :, self.in_context_len :
                                ]
                            else:
                                repa_hidden_states[i] = x_added
                        layer_idx += 1
                    # Both streams get the same fused representation for post-blocks
                    x_pixel = x_added
                    x_dino = x_added
                else:
                    for i, block in enumerate(self.shared_blocks):
                        if (
                            self.in_context_len > 0
                            and layer_idx >= self.in_context_start
                            and not in_context_added
                        ):
                            in_context_tokens = y_emb.unsqueeze(1).repeat(
                                1, self.in_context_len, 1
                            )
                            in_context_tokens += self.in_context_posemb
                            x_pixel = torch.cat([in_context_tokens, x_pixel], dim=1)
                            x_dino = torch.cat(
                                [in_context_tokens.clone(), x_dino], dim=1
                            )
                            in_context_added = True
                        if in_context_added:
                            rope = self.feat_rope_incontext
                        x_pixel = block(x_pixel, c, rope)
                        x_dino = block(x_dino, c, rope)

                        # Collect hidden states for REPA (from pixel stream, excluding in-context tokens)
                        if repa_hidden_states is not None and i in self.repa_layers:
                            if in_context_added:
                                repa_hidden_states[i] = x_pixel[
                                    :, self.in_context_len :
                                ]
                            else:
                                repa_hidden_states[i] = x_pixel
                        layer_idx += 1

                # Post-processing: K separate blocks for each stream
                for i, (pixel_block, dino_block) in enumerate(
                    zip(self.pixel_post_blocks, self.dino_post_blocks)
                ):
                    if (
                        self.in_context_len > 0
                        and layer_idx >= self.in_context_start
                        and not in_context_added
                    ):
                        in_context_tokens = y_emb.unsqueeze(1).repeat(
                            1, self.in_context_len, 1
                        )
                        in_context_tokens += self.in_context_posemb
                        x_pixel = torch.cat([in_context_tokens, x_pixel], dim=1)
                        x_dino = torch.cat([in_context_tokens.clone(), x_dino], dim=1)
                        in_context_added = True
                    if in_context_added:
                        rope = self.feat_rope_incontext
                    x_pixel = pixel_block(x_pixel, c, rope)
                    x_dino = dino_block(x_dino, c, rope)
                    layer_idx += 1

                # Remove in-context tokens before final layer
                if self.in_context_len > 0 and in_context_added:
                    x_pixel = x_pixel[:, self.in_context_len :]
                    x_dino = x_dino[:, self.in_context_len :]

                # Final layer
                x = self.final_layer(x_pixel, c, x_dino=x_dino)

        elif self.use_mmdit:
            x_pixel, x_dino = self.x_embedder(x, zs=zs, t_emb=embedder_t_emb)
            x_pixel += self.pos_embed
            x_dino += self.pos_embed

            for i, block in enumerate(self.blocks):
                rope = self.feat_rope
                if self.in_context_len > 0 and i >= self.in_context_start:
                    if i == self.in_context_start:
                        in_context_tokens = y_emb.unsqueeze(1).repeat(
                            1, self.in_context_len, 1
                        )
                        in_context_tokens += self.in_context_posemb
                        x_pixel = torch.cat([in_context_tokens, x_pixel], dim=1)
                        x_dino = torch.cat([in_context_tokens.clone(), x_dino], dim=1)
                    rope = self.feat_rope_incontext
                # Process pixel and dino streams through MMDiT block
                x_pixel, x_dino = block(
                    x_pixel,
                    x_dino,
                    c,
                    rope,
                    block_dino_to_pixel=block_dino_to_pixel,
                    symmetric_attn_mask=symmetric_attn_mask,
                )

                # Collect hidden states for REPA (after block, excluding in-context tokens)
                if repa_hidden_states is not None and i in self.repa_layers:
                    if self.in_context_len > 0 and i >= self.in_context_start:
                        repa_hidden_states[i] = x_pixel[:, self.in_context_len :]
                    else:
                        repa_hidden_states[i] = x_pixel

            if self.in_context_len > 0:
                x_pixel = x_pixel[:, self.in_context_len :]
                x_dino = x_dino[:, self.in_context_len :]

            x = self.final_layer(x_pixel, c, x_dino=x_dino)
        else:
            # forward JiT (single-stream or pixel-only)
            if self.use_dinov2 and self.use_co_embed:
                # Co-denoising: embedder returns (pixel, dino) tuple
                x_pixel, x_dino = self.x_embedder(x, zs=zs, t_emb=embedder_t_emb)

                if x_dino is None:
                    x = x_pixel + self.pos_embed
                    use_concat_rope = False
                else:
                    # Add positional embeddings
                    x_pixel = x_pixel + self.pos_embed
                    x_dino = x_dino + self.pos_embed

                    # Concatenate along sequence dimension for single-stream processing
                    N_pixel = x_pixel.shape[1]
                    N_dino = x_dino.shape[1]
                    x = torch.cat([x_pixel, x_dino], dim=1)  # (B, N_pixel + N_dino, D)
                    use_concat_rope = True
            else:
                # Pixel-only mode: embedder returns single tensor
                x = self.x_embedder(x, zs=zs, t_emb=embedder_t_emb)
                x = x + self.pos_embed
                x_dino = None
                use_concat_rope = False

            for i, block in enumerate(self.blocks):
                # in-context
                if self.in_context_len > 0 and i == self.in_context_start:
                    in_context_tokens = y_emb.unsqueeze(1).repeat(
                        1, self.in_context_len, 1
                    )
                    in_context_tokens += self.in_context_posemb
                    x = torch.cat([in_context_tokens, x], dim=1)

                # Select appropriate RoPE based on whether we have concatenated sequence
                if use_concat_rope:
                    rope = (
                        self.feat_rope_concat
                        if i < self.in_context_start
                        else self.feat_rope_concat_incontext
                    )
                else:
                    rope = (
                        self.feat_rope
                        if i < self.in_context_start
                        else self.feat_rope_incontext
                    )

                x = block(x, c, rope)

                # Collect hidden states for REPA (after block, excluding in-context tokens)
                if repa_hidden_states is not None and i in self.repa_layers:
                    if self.in_context_len > 0 and i >= self.in_context_start:
                        repa_hidden_states[i] = x[:, self.in_context_len :]
                    else:
                        repa_hidden_states[i] = x

            x = x[:, self.in_context_len :]

            if x_dino is not None:
                # Split back into pixel and dino streams before final layer
                x_pixel, x_dino = torch.split(x, [N_pixel, N_dino], dim=1)
                x = self.final_layer(x_pixel, c, x_dino=x_dino)
            else:
                x = self.final_layer(x, c)

        if self.use_co_embed:
            if inference_pixel_only and self.use_shared_jit:
                # SharedJiT inference: x is already just pixel output from FinalLayer
                # unpatchify for pixel only (don't split into pixel/dinov2)
                c = self.out_channels
                p = self.patch_size
                h = w = int(x.shape[1] ** 0.5)
                x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
                x = torch.einsum("nhwpqc->nchpwq", x)
                output = x.reshape(shape=(x.shape[0], c, h * p, h * p))
                return output
            else:
                # Training or non-SharedJiT: x contains both pixel and dinov2 concatenated
                pixel_output, dinov2_output = self.unpatchify(x, self.patch_size)
                if repa_hidden_states is not None:
                    return pixel_output, dinov2_output, repa_hidden_states
                return pixel_output, dinov2_output
        else:
            output = self.unpatchify(x, self.patch_size)
            if repa_hidden_states is not None:
                return output, repa_hidden_states
            if return_hidden_states:
                return {"output": output, "hidden_states": hidden_states}
            return output


def JiT_B_16(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_mlp_ratio_8(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        mlp_ratio=8,
        **kwargs,
    )


def JiT_B_16_mlp_ratio_12(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        mlp_ratio=12,
        **kwargs,
    )


def JiT_B_16_mlp_ratio_14(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        mlp_ratio=14,
        **kwargs,
    )


def JiT_B_16_co(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_co_depth10(**kwargs):
    return JiT(
        depth=10,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_co_depth8(**kwargs):
    return JiT(
        depth=8,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_co_depth6(**kwargs):
    return JiT(
        depth=6,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_co_depth4(**kwargs):
    return JiT(
        depth=4,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_co_depth2(**kwargs):
    return JiT(
        depth=2,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_hidden_size_1024(**kwargs):
    return JiT(
        depth=12,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_hidden_size_1280(**kwargs):
    return JiT(
        depth=12,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_hidden_size_1152(**kwargs):
    return JiT(
        depth=12,
        hidden_size=1152,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_16_hidden_size_1088(**kwargs):
    return JiT(
        depth=12,
        hidden_size=1088,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_32(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=32,
        **kwargs,
    )


def JiT_L_16(**kwargs):
    return JiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
        **kwargs,
    )


def JiT_L_16_co(**kwargs):
    return JiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
        **kwargs,
    )


def JiT_L_32(**kwargs):
    return JiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=32,
        **kwargs,
    )


def JiT_H_16(**kwargs):
    return JiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
        **kwargs,
    )


def JiT_G_16(**kwargs):
    return JiT(
        depth=40,
        hidden_size=1664,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
        **kwargs,
    )


def JiT_H_16_co(**kwargs):
    return JiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
        **kwargs,
    )


def JiT_H_32(**kwargs):
    return JiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=32,
        **kwargs,
    )


JiT_models = {
    "JiT-B/16": JiT_B_16,
    "JiT-B/32": JiT_B_32,
    "JiT-L/16": JiT_L_16,
    "JiT-L/32": JiT_L_32,
    "JiT-H/16": JiT_H_16,
    "JiT-H/32": JiT_H_32,
    "JiT-G/16": JiT_G_16,
    "JiT-B/16-co": JiT_B_16_co,
    "JiT-B/16-co-depth10": JiT_B_16_co_depth10,
    "JiT-B/16-co-depth8": JiT_B_16_co_depth8,
    "JiT-B/16-co-depth6": JiT_B_16_co_depth6,
    "JiT-B/16-co-depth4": JiT_B_16_co_depth4,
    "JiT-B/16-co-depth2": JiT_B_16_co_depth2,
    "JiT-L/16-co": JiT_L_16_co,
    "JiT-H/16-co": JiT_H_16_co,
    "JiT_B/16-hidden_size_1024": JiT_B_16_hidden_size_1024,
    "JiT_B/16-hidden_size_1280": JiT_B_16_hidden_size_1280,
    "JiT_B/16-hidden_size_1152": JiT_B_16_hidden_size_1152,
    "JiT_B/16-hidden_size_1088": JiT_B_16_hidden_size_1088,
    "JiT-B/16-mlp_ratio_8": JiT_B_16_mlp_ratio_8,
    "JiT-B/16-mlp_ratio_12": JiT_B_16_mlp_ratio_12,
    "JiT-B/16-mlp_ratio_14": JiT_B_16_mlp_ratio_14,
}
