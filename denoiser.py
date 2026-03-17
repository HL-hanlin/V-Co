import projection_loss as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model_cot import JiTCoT_models
from model_vco import JiT_models

try:
    from transformers import AutoModel

    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_dinov2 = getattr(args, "use_dinov2", False)
        self.use_co_embed = getattr(args, "use_co_embed", False)
        self.dinov2_loss_coef = getattr(args, "dinov2_loss_coef", 1.0)
        self.pixels_loss_coef = getattr(args, "pixels_loss_coef", 1.0)

        # REPA (Representation Alignment) configuration - Official iREPA implementation
        self.enable_repa = getattr(args, "enable_repa", False)
        self.encoder_depth = getattr(args, "encoder_depth", 8)
        self.projector_dim = getattr(args, "projector_dim", 2048)
        self.projection_layer_type = getattr(args, "projection_layer_type", "mlp")
        self.proj_kwargs_kernel_size = getattr(args, "proj_kwargs_kernel_size", 3)
        self.spnorm_method = getattr(args, "spnorm_method", "none")
        self.zscore_alpha = getattr(args, "zscore_alpha", 0.8)
        self.use_jit_cot_model = getattr(args, "use_jit_cot_model", False)

        # Parse projection loss types and coefficients
        projection_loss_type_str = getattr(args, "projection_loss_type", "cosine")
        self.projection_loss_type = [
            elem.strip() for elem in projection_loss_type_str.split(",") if elem.strip()
        ]
        proj_coeff_str = getattr(args, "proj_coeff", "0.5")
        self.proj_coeff = [
            float(elem.strip()) for elem in proj_coeff_str.split(",") if elem.strip()
        ]

        assert len(self.projection_loss_type) == len(
            self.proj_coeff
        ), f"len(self.projection_loss_type) - {len(self.projection_loss_type)} != len(self.proj_coeff) - {len(self.proj_coeff)}"

        # Create projection losses
        if self.enable_repa:
            self.projection_loss = [
                pl.make_projection_loss(loss_type)
                for loss_type in self.projection_loss_type
            ]
            # Parse repa_layers from encoder_depth (single layer for official iREPA pattern)
            self.repa_layers = [self.encoder_depth]
        else:
            self.projection_loss = []
            self.repa_layers = None

        # Determine freeze settings
        freeze_dinov2 = not getattr(args, "unfreeze_dinov2", False)

        if self.use_jit_cot_model:
            self.net = JiTCoT_models[args.model](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                bottleneck_dim_dino=128,
                dh_depth=args.dh_depth,
                dh_hidden_size=768,
                dino_in_channels=768,
                use_channel_concat=getattr(args, "use_channel_concat", False),
                use_token_concat=getattr(args, "use_token_concat", False),
                match_pixel_norm=getattr(args, "match_pixel_norm", 0.485),
            )

        else:
            self.net = JiT_models[args.model](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                use_dinov2=self.use_dinov2,
                dinov2_model_name=getattr(
                    args, "dinov2_model_name", "facebook/dinov2-with-registers-base"
                ),
                freeze_dinov2=freeze_dinov2,
                use_co_embed=args.use_co_embed,
                use_gated_co_embed=args.use_gated_co_embed,
                use_mmdit=getattr(args, "use_mmdit", False),
                separate_qkv=getattr(args, "separate_qkv", False),
                jit_dino_proj=getattr(args, "jit_dino_proj", False),
                jit_pixel_proj=getattr(args, "jit_pixel_proj", False),
                jit_dino_head=getattr(args, "jit_dino_head", False),
                jit_pixel_head=getattr(args, "jit_pixel_head", False),
                jit_refiner_layers=getattr(args, "jit_refiner_layers", 2),
                use_shared_jit=getattr(args, "use_shared_jit", False),
                use_channel_concat=getattr(args, "use_channel_concat", False),
                use_direct_addition=getattr(args, "use_direct_addition", False),
                use_conv2d_dino_proj=getattr(args, "use_conv2d_dino_proj", False),
                use_dino_from_rae=getattr(args, "use_dino_from_rae", False),
                match_pixel_norm=getattr(args, "match_pixel_norm", 0.485),
            )

        # Store SharedJiT flag for inference
        self.use_shared_jit = getattr(args, "use_shared_jit", False)

        # Initialize REPA projectors if enabled
        if self.enable_repa and self.repa_layers is not None:
            # DINOv2 feature dimension is 768 for base model
            self.net.init_repa_projectors(self.repa_layers, target_dim=768)

        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.dinov2_drop_prob = getattr(args, "dinov2_drop_prob", 0.0)
        self.label_dinov2_drop_prob = getattr(args, "label_dinov2_drop_prob", 0.0)
        self.dinov2_drop_zero_loss = getattr(args, "dinov2_drop_zero_loss", False)
        self.dinov2_null_type = getattr(args, "dinov2_null_type", "zero")

        # Learnable null DINOv2 token for CFG (when dinov2_null_type == 'learned')
        if self.dinov2_null_type == "learned":
            dinov2_dim = 768  # DINOv2 base feature dim
            self.dino_null_token = nn.Parameter(torch.zeros(1, 1, dinov2_dim))
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        # Separate noise scale for DINOv2 features (defaults to same as pixels)
        self.noise_scale_dinov2 = getattr(args, "noise_scale_dinov2", None)
        self.use_dino_time_shift = getattr(args, "use_dino_time_shift", False)
        self.dino_time_shift_alpha = getattr(args, "dino_time_shift_alpha", 0.485)
        if self.noise_scale_dinov2 is None:
            self.noise_scale_dinov2 = self.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_label = args.cfg
        self.cfg_dino = getattr(args, "cfg_dino", None)
        self.cfg_interval = (args.interval_min, args.interval_max)

        # Auxiliary DINOv2 loss (default to disabled)
        self.aux_dinov2_loss = getattr(args, "aux_dinov2_loss", False)
        self.aux_dinov2_loss_coeff = getattr(args, "aux_dinov2_loss_coeff", 1.0)
        self.aux_dinov2_low_sim_gate = getattr(args, "aux_dinov2_low_sim_gate", False)
        self.aux_dinov2_low_sim_thresh = getattr(args, "aux_dinov2_low_sim_thresh", 0.5)
        self.class_balanced_sampling = getattr(args, "class_balanced_sampling", False)
        self.use_dino_from_rae = getattr(args, "use_dino_from_rae", False)

        # DINOv2 specific settings
        if self.use_dinov2 or self.use_co_embed:
            self.dinov2_input_size = 224  # DINOv2 expects 224x224
            # ImageNet normalization for DINOv2
            self.register_buffer(
                "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        # Auxiliary DINOv2 loss: create a frozen DINOv2 model for computing perceptual loss
        # This is outside the use_dinov2/use_co_embed block so it can work independently
        if self.aux_dinov2_loss:
            # Ensure we have the necessary DINOv2 settings for auxiliary loss
            if not hasattr(self, "dinov2_input_size"):
                self.dinov2_input_size = 224  # DINOv2 expects 224x224
                self.register_buffer(
                    "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                )
                self.register_buffer(
                    "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                )
            # Create a separate frozen DINOv2 model for auxiliary loss
            # We need gradients to flow through for training the main model,
            # but the DINOv2 parameters themselves should not be updated
            self.aux_dinov2 = AutoModel.from_pretrained(args.dinov2_model_name)
            # Freeze all parameters (non-trainable)
            for param in self.aux_dinov2.parameters():
                param.requires_grad = False
            # But we still need to allow gradients to flow through for backprop
            # This is done by NOT using torch.no_grad() when computing the loss

        # Drifting V3 loss (Design B: gated attraction + repulsion from generated neighbors)
        self.drifting_v3_loss = getattr(args, "drifting_v3_loss", False)
        self.drifting_v3_loss_coef = getattr(args, "drifting_v3_loss_coef", 1.0)
        self.drifting_v3_feat_type = getattr(args, "drifting_v3_feat_type", "cls")
        self.drifting_v3_gate_tau = getattr(args, "drifting_v3_gate_tau", 1.0)
        self.use_fixed_gate_tau = getattr(args, "use_fixed_gate_tau", False)
        self.drifting_v3_repulsion_tau = getattr(args, "drifting_v3_repulsion_tau", 1.0)
        self.drifting_v3_w_attract = getattr(args, "drifting_v3_w_attract", 1.0)
        self.drifting_v3_w_repel = getattr(args, "drifting_v3_w_repel", 1.0)
        self.uncond_dino_null = getattr(args, "uncond_dino_null", False)

        if self.drifting_v3_loss:
            if not hasattr(self, "dinov2_input_size"):
                self.dinov2_input_size = 224
                self.register_buffer(
                    "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                )
                self.register_buffer(
                    "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                )
            # Reuse existing DINOv2 model if available
            if hasattr(self, "drift_dinov2"):
                self.drifting_dinov2 = self.drift_dinov2
            elif hasattr(self, "aux_dinov2"):
                self.drifting_dinov2 = self.aux_dinov2
            elif hasattr(self, "vec_dinov2"):
                self.drifting_dinov2 = self.vec_dinov2
            else:
                self.drifting_dinov2 = AutoModel.from_pretrained(args.dinov2_model_name)
                for param in self.drifting_dinov2.parameters():
                    param.requires_grad = False

        self.dinov2_model_name = getattr(
            args, "dinov2_model_name", "facebook/dinov2-with-registers-base"
        )
        if "giant" in self.dinov2_model_name:
            self.dinov2_dim = 1536
        elif "large" in self.dinov2_model_name:
            self.dinov2_dim = 1024
        elif "small" in self.dinov2_model_name:
            self.dinov2_dim = 384
        else:
            # default/base
            self.dinov2_dim = 768

    def normalize_for_dinov2(self, x):
        """Convert from [-1, 1] to ImageNet normalized space."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = (x - self.img_mean) / self.img_std
        return x

    def denormalize_from_dinov2(self, x):
        """Convert from ImageNet normalized space back to [-1, 1]."""
        x = x * self.img_std + self.img_mean  # ImageNet -> [0, 1]
        x = x * 2 - 1  # [0, 1] -> [-1, 1]
        return x

    def resize_for_dinov2(self, x):
        """Resize from img_size (256) to DINOv2 input size (224)."""
        if x.shape[-1] != self.dinov2_input_size:
            x = F.interpolate(
                x,
                size=(self.dinov2_input_size, self.dinov2_input_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def _extract_drifting_v3_features(self, model_out):
        """Extract features for drifting V3 loss based on self.drifting_v3_feat_type."""
        if self.drifting_v3_feat_type == "avg_pool2d":
            num_special_tokens = 1 + 4  # CLS + 4 registers
            patch_tokens = model_out.last_hidden_state[:, num_special_tokens:, :]
            B, N, D = patch_tokens.shape
            h = w = int(N**0.5)
            spatial = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)
            return F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        else:
            return model_out.last_hidden_state[:, 0, :]  # CLS token

    def compute_drifting_v3_loss(self, x_gen, labels_gen, x_pos, labels_pos):
        """
        Drifting V3 loss (Design B): Gated attraction to paired GT +
        repulsion from other same-class generated neighbors in DINOv2 feature space.

        For each sample i in class c:
          - s_i = exp(-||feat_gen_i - feat_pos_i||^2 / gate_tau)  (similarity gate)
          - attraction = w_attract * s_i * (feat_pos_i - feat_gen_i)
          - repulsion = w_repel * (1-s_i) * (1/|M_i|) * sum_j kernel_j * (feat_gen_j - feat_gen_i)
            where j != i, class(j) = class(i), kernel_j = exp(-||feat_gen_i - feat_gen_j||^2 / repulsion_tau)
          - V_i = attraction - repulsion
          - loss_i = MSE(feat_gen_i, stopgrad(feat_gen_i + V_i))

        When z is far from target: s_i ~ 0 => repulsion ON, prevents mode collapse
        When z is close to target: s_i ~ 1 => pure attraction, clean convergence

        Returns:
            loss_v3: Scalar loss
            info: Dict with monitoring values
        """

        device = x_gen.device

        # DINOv2 forward pass
        x_gen_224 = self.resize_for_dinov2(x_gen)
        x_gen_224_norm = self.normalize_for_dinov2(x_gen_224)
        x_pos_224 = self.resize_for_dinov2(x_pos)
        x_pos_224_norm = self.normalize_for_dinov2(x_pos_224)

        gen_out = self.drifting_dinov2(x_gen_224_norm, return_dict=True)
        with torch.no_grad():
            pos_out = self.drifting_dinov2(x_pos_224_norm, return_dict=True)

        feat_gen = self._extract_drifting_v3_features(gen_out).float()
        feat_pos = self._extract_drifting_v3_features(pos_out).float()

        # L2 normalize
        feat_gen = F.normalize(feat_gen, dim=-1)
        feat_pos = F.normalize(feat_pos, dim=-1)

        unique_labels = torch.unique(labels_gen)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_attract_norm = 0.0
        total_repel_norm = 0.0
        total_v_norm = 0.0
        total_gate_mean = 0.0
        num_samples = 0

        for c in unique_labels:
            mask_gen = labels_gen == c
            mask_pos = labels_pos == c

            if not mask_pos.any():
                continue

            feat_gen_c = feat_gen[mask_gen]  # (Nc, D)
            feat_pos_c = feat_pos[mask_pos]  # (Nc, D)
            Nc = feat_gen_c.shape[0]

            if Nc < 2:
                # Need at least 2 samples for repulsion; with 1 sample, just do attraction
                if self.use_fixed_gate_tau:
                    s_i = torch.full_like(
                        feat_gen_c[:, :1], self.drifting_v3_gate_tau
                    )  # (Nc, 1)
                else:
                    s_i = torch.exp(
                        -((feat_gen_c - feat_pos_c) ** 2).sum(dim=-1, keepdim=True)
                        / self.drifting_v3_gate_tau
                    )
                V_i = self.drifting_v3_w_attract * s_i * (feat_pos_c - feat_gen_c)
                target = (feat_gen_c + V_i).detach()
                loss_c = F.mse_loss(feat_gen_c, target)
                total_loss = total_loss + loss_c
                total_v_norm += (V_i**2).mean().item() ** 0.5
                total_gate_mean += s_i.mean().item()
                num_samples += 1
                continue

            # Pairwise distances between generated features: (Nc, Nc)
            dist_gen_gen = torch.cdist(feat_gen_c, feat_gen_c, p=2) ** 2

            # Distance from each generated to its paired positive: (Nc,)
            dist_to_pos = ((feat_gen_c - feat_pos_c) ** 2).sum(dim=-1)

            # Similarity gate
            if self.use_fixed_gate_tau:
                s_i = torch.full(
                    (Nc,),
                    self.drifting_v3_gate_tau,
                    device=device,
                    dtype=feat_gen_c.dtype,
                )
                print(s_i)
            else:
                # s_i = exp(-||feat_gen_i - feat_pos_i||^2 / gate_tau)
                s_i = torch.exp(-dist_to_pos / self.drifting_v3_gate_tau)  # (Nc,)

            # Attraction: w+ * s_i * (feat_pos_i - feat_gen_i)
            attraction = (
                self.drifting_v3_w_attract
                * s_i.unsqueeze(-1)
                * (feat_pos_c - feat_gen_c)
            )  # (Nc, D)

            # Repulsion kernel: exp(-||feat_gen_i - feat_gen_j||^2 / repulsion_tau)
            repulsion_kernel = torch.exp(
                -dist_gen_gen / self.drifting_v3_repulsion_tau
            )  # (Nc, Nc)
            # Mask out self
            repulsion_kernel = repulsion_kernel * (1.0 - torch.eye(Nc, device=device))
            # Normalize per row
            kernel_sum = repulsion_kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            repulsion_weights = repulsion_kernel / kernel_sum  # (Nc, Nc)
            # print(repulsion_weights)

            # Weighted sum of (feat_gen_j - feat_gen_i) for repulsion
            # repulsion_weights @ feat_gen_c gives weighted sum of feat_gen_j
            weighted_neighbors = repulsion_weights @ feat_gen_c  # (Nc, D)
            repulsion_dir = (
                weighted_neighbors - feat_gen_c
            )  # (Nc, D) points from i toward neighbors

            # Repulsion: w- * (1-s_i) * repulsion_dir (we subtract this, so V pushes AWAY from neighbors)
            repulsion = (
                self.drifting_v3_w_repel * (1.0 - s_i).unsqueeze(-1) * repulsion_dir
            )  # (Nc, D)

            # V = attraction - repulsion
            V_i = attraction - repulsion  # (Nc, D)

            target = (feat_gen_c + V_i).detach()
            loss_c = F.mse_loss(feat_gen_c, target)

            total_loss = total_loss + loss_c
            total_attract_norm += (attraction**2).mean().item() ** 0.5
            total_repel_norm += (repulsion**2).mean().item() ** 0.5
            total_v_norm += (V_i**2).mean().item() ** 0.5
            total_gate_mean += s_i.mean().item()
            num_samples += 1

        if num_samples > 0:
            loss_v3 = total_loss / num_samples
            info_v3 = {
                "drifting_v3_loss": loss_v3.item(),
                "drifting_v3_drift_norm": total_v_norm / num_samples,
                "drifting_v3_attract_norm": total_attract_norm / num_samples,
                "drifting_v3_repel_norm": total_repel_norm / num_samples,
                "drifting_v3_gate_mean": total_gate_mean / num_samples,
            }
        else:
            loss_v3 = torch.tensor(0.0, device=device, requires_grad=True)
            info_v3 = {
                "drifting_v3_loss": 0.0,
                "drifting_v3_drift_norm": 0.0,
                "drifting_v3_attract_norm": 0.0,
                "drifting_v3_repel_norm": 0.0,
                "drifting_v3_gate_mean": 0.0,
            }

        return loss_v3, info_v3

    def resize_from_dinov2(self, x):
        """Resize from DINOv2 output size back to img_size (256)."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def drop_labels_and_dinov2(self, labels, z_dinov2):
        """
        Jointly drop BOTH labels and DINOv2 with a single Bernoulli draw.
        Either both are dropped or neither is dropped.
        """
        B = labels.shape[0]
        drop = torch.rand(B, device=labels.device) < self.label_dinov2_drop_prob

        # Drop labels: replace with null class
        labels = torch.where(drop, torch.full_like(labels, self.num_classes), labels)

        # Drop DINOv2: replace with null condition
        if drop.any():
            N, D = z_dinov2.shape[1], z_dinov2.shape[2]
            if self.dinov2_null_type == "learned":
                null_cond = self.dino_null_token.to(z_dinov2.device).expand(B, N, -1)
            else:
                null_cond = torch.zeros_like(z_dinov2)

            drop_expanded = drop[:, None, None].expand_as(z_dinov2)
            z_dinov2 = torch.where(drop_expanded, null_cond, z_dinov2)

        return labels, z_dinov2

    def drop_dinov2(self, z_dinov2):
        """
        Drop DINOv2 condition for CFG by replacing with null condition.
        z_dinov2: (B, N, D) noisy DINOv2 latents
        Returns: z_dinov2 with some samples replaced by null, and the drop mask

        Null condition types:
        - 'zero': Use zeros (appropriate when standardize_dinov2=True, where zero is the mean)
        - 'mean': Use pre-computed mean from dinov2_latent_mean (before standardization)
        - 'noise': Use pure noise at t=0 (noise_scale_dinov2 * randn)
        - 'learned': Use learnable null token
        """
        if self.dinov2_drop_prob <= 0:
            return z_dinov2, None

        B, N, D = z_dinov2.shape
        drop = torch.rand(B, device=z_dinov2.device) < self.dinov2_drop_prob
        # Expand mask to match z_dinov2 shape: (B,) -> (B, 1, 1)
        drop_expanded = drop.view(B, 1, 1)

        # Determine null condition based on type
        if self.dinov2_null_type == "zero":
            null_cond = torch.zeros_like(z_dinov2)
        elif self.dinov2_null_type == "learned":
            null_cond = self.dino_null_token.to(z_dinov2.device).expand(B, N, -1)
        else:
            null_cond = torch.zeros_like(z_dinov2)

        out = torch.where(drop_expanded, null_cond, z_dinov2)
        return out, drop

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, repa_target_features=None):
        """
        Training forward pass.
        x: input images in [-1, 1] at img_size (256x256)
        repa_target_features: optional external encoder features for REPA loss (used when enable_repa=True and use_co_embed=False)

        self = model.module
        """
        if self.training and self.label_dinov2_drop_prob > 0:
            # Joint dropping: single Bernoulli draw, both dropped or neither
            assert (
                self.label_drop_prob == 0 and self.dinov2_drop_prob == 0
            ), "When using --label_dinov2_drop_prob, set --label_drop_prob 0 and --dinov2_drop_prob 0"
            joint_drop_mask = (
                torch.rand(labels.shape[0], device=labels.device)
                < self.label_dinov2_drop_prob
            ).to(labels.device)
            labels_dropped = torch.where(
                joint_drop_mask,
                torch.full_like(labels, self.num_classes),
                labels,
            )
        else:
            joint_drop_mask = None
            labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))

        return_hidden_states = hidden_state_layers = None

        if self.use_dinov2 and self.use_co_embed:

            if self.use_dino_time_shift:
                alpha = self.dino_time_shift_alpha
                t_dinov2 = alpha * t / (1 + (alpha - 1) * t)
            else:
                t_dinov2 = t

            ### 1. pixels
            e = torch.randn_like(x) * self.noise_scale
            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(self.t_eps)

            ### 2. dinov2 features (current version is to first encode with dinov2, then add noise)
            with torch.no_grad():
                patch_features = self.net.x_embedder.dinov2.encode(x * 0.5 + 0.5)
            patch_features = rearrange(patch_features, "b c h w -> b (h w) c")

            # Compute SNR metrics for monitoring
            # Use RMS (root mean square) for consistent signal power measurement
            pixel_rms = torch.sqrt((x**2).mean())
            dino_rms = torch.sqrt((patch_features**2).mean())
            pixel_snr = pixel_rms / self.noise_scale
            dino_snr = dino_rms / self.noise_scale_dinov2
            snr_ratio = dino_snr / pixel_snr

            # Diagnostic logging (optional - can remove after verification)
            if self.training and torch.rand(1).item() < 0.01:
                print(
                    f"[SNR Check] Pixel: {pixel_snr:.3f} | DINOv2: {dino_snr:.3f} | Ratio: {snr_ratio:.3f} | "
                )

            # add noise (use t_dinov2 which may be independent from t)
            # Independent noise for DINOv2 (original behavior)
            e_dinov2 = torch.randn_like(patch_features) * self.noise_scale_dinov2
            # Flow matching in normalized feature space
            z_dinov2 = (
                t_dinov2.squeeze(-1) * patch_features
                + (1 - t_dinov2).squeeze(-1) * e_dinov2
            )
            v_dinov2 = (patch_features - z_dinov2) / (1 - t_dinov2).squeeze(
                -1
            ).clamp_min(self.t_eps)

            # Drop DINOv2 condition for CFG
            block_dino_to_pixel_mask = None  # attention mask mode: (B,) bool tensor
            dino_drop_mask = None  # tracks which samples had DINOv2 dropped (for dinov2_drop_zero_loss)
            use_attn_mask = self.dinov2_null_type in (
                "attn_mask_asymmetric",
                "attn_mask_symmetric",
            )
            symmetric_attn_mask = self.dinov2_null_type == "attn_mask_symmetric"
            if use_attn_mask:
                # Attention mask mode: keep real DINOv2 tokens, block via attention mask
                z_dinov2_dropped = z_dinov2
                if joint_drop_mask is not None:
                    block_dino_to_pixel_mask = joint_drop_mask
                    dino_drop_mask = joint_drop_mask
                elif self.training and self.dinov2_drop_prob > 0:
                    block_dino_to_pixel_mask = (
                        torch.rand(z_dinov2.shape[0], device=z_dinov2.device)
                        < self.dinov2_drop_prob
                    )
                    dino_drop_mask = block_dino_to_pixel_mask
            elif joint_drop_mask is not None:
                # Joint dropping mode: apply same mask used for labels
                B_jd, N_jd, D_jd = z_dinov2.shape
                if self.dinov2_null_type == "learned":
                    null_cond_jd = self.dino_null_token.to(z_dinov2.device).expand(
                        B_jd, N_jd, -1
                    )
                else:
                    null_cond_jd = torch.zeros_like(z_dinov2)
                drop_expanded_jd = joint_drop_mask[:, None, None].expand_as(z_dinov2)
                z_dinov2_dropped = torch.where(drop_expanded_jd, null_cond_jd, z_dinov2)
                dino_drop_mask = joint_drop_mask
            elif self.training and self.dinov2_drop_prob > 0:
                # Independent dropping mode
                z_dinov2_dropped, dino_drop_mask = self.drop_dinov2(z_dinov2)
            else:
                z_dinov2_dropped = z_dinov2

            # Forward pass with REPA hidden states if enabled
            if self.enable_repa:
                net_pixels_out, net_dinov2_out, repa_hidden_states = self.net(
                    z,
                    t.flatten(),
                    labels_dropped,
                    return_hidden_states=return_hidden_states,
                    hidden_state_layers=hidden_state_layers,
                    zs=z_dinov2_dropped,
                    return_repa_hidden_states=True,
                    block_dino_to_pixel=block_dino_to_pixel_mask,
                    symmetric_attn_mask=symmetric_attn_mask,
                )
            else:
                net_pixels_out, net_dinov2_out = self.net(
                    z,
                    t.flatten(),
                    labels_dropped,
                    return_hidden_states=return_hidden_states,
                    hidden_state_layers=hidden_state_layers,
                    zs=z_dinov2_dropped,
                    block_dino_to_pixel=block_dino_to_pixel_mask,
                    symmetric_attn_mask=symmetric_attn_mask,
                )
                repa_hidden_states = None

            ### 1. pixels
            x_pixels_pred = net_pixels_out
            v_pixels_pred = (x_pixels_pred - z) / (1 - t).clamp_min(self.t_eps)
            loss_pixels = (v - v_pixels_pred) ** 2

            ### 2. dinov2 features
            x_dinov2_pred = net_dinov2_out
            v_dinov2_pred = (x_dinov2_pred - z_dinov2) / (1 - t.squeeze(-1)).clamp_min(
                self.t_eps
            )
            # v_dinov2_pred = (x_dinov2_pred - z_dinov2) / (1 - t).clamp_min(self.t_eps)
            loss_dinov2 = (v_dinov2 - v_dinov2_pred) ** 2

            # Zero out dinov2 loss for samples where DINOv2 was dropped (CFG dropping)
            if self.dinov2_drop_zero_loss and dino_drop_mask is not None:
                loss_dinov2[dino_drop_mask] = 0.0

            ### 3. REPA loss (align pixel stream hidden states with clean DINOv2 features)
            loss_repa = torch.tensor(0.0, device=x.device)
            if (
                self.enable_repa
                and repa_hidden_states is not None
                and len(repa_hidden_states) > 0
            ):
                # Use official iREPA pattern with projection losses
                total_proj_loss = torch.tensor(0.0, device=x.device)
                for proj_loss_name, proj_loss_fn, coeff in zip(
                    self.projection_loss_type, self.projection_loss, self.proj_coeff
                ):
                    proj_loss = torch.tensor(0.0, device=x.device)
                    for layer_idx, hidden in repa_hidden_states.items():
                        # Project hidden states to target feature space
                        projected = self.net.repa_projectors[str(layer_idx)](hidden)
                        # Compute projection loss with clean DINOv2 features
                        proj_loss = proj_loss + proj_loss_fn(
                            patch_features.detach(), projected
                        )
                    proj_loss = proj_loss / len(repa_hidden_states)
                    total_proj_loss = total_proj_loss + coeff * proj_loss
                loss_repa = total_proj_loss

            ### 4. Auxiliary DINOv2 loss: 1 - cosine_similarity(DINOv2(x_gt), DINOv2(x_pred))
            loss_aux_dinov2 = torch.tensor(0.0, device=x.device)
            if self.aux_dinov2_loss:
                # Get the predicted clean image from the pixel branch
                # x_pixels_pred is the model's prediction of the clean image
                # x is the ground truth clean image

                # Resize and normalize images for DINOv2 (no gradient blocking for x_pixels_pred)
                x_gt_224 = self.resize_for_dinov2(x)
                x_gt_224_normalized = self.normalize_for_dinov2(x_gt_224)

                x_pred_224 = self.resize_for_dinov2(x_pixels_pred)
                x_pred_224_normalized = self.normalize_for_dinov2(x_pred_224)

                # Extract DINOv2 features (frozen model, but gradients flow through input)
                # Note: we DON'T use torch.no_grad() here so gradients can flow back
                aux_dinov2_out_gt = self.aux_dinov2(
                    x_gt_224_normalized, return_dict=True
                )
                aux_dinov2_out_pred = self.aux_dinov2(
                    x_pred_224_normalized, return_dict=True
                )

                # Get patch features (exclude CLS and register tokens)
                num_special_tokens = 1 + 4  # CLS + 4 registers
                feat_gt = aux_dinov2_out_gt.last_hidden_state[:, num_special_tokens:, :]
                feat_pred = aux_dinov2_out_pred.last_hidden_state[
                    :, num_special_tokens:, :
                ]

                # Compute cosine similarity loss: 1 - cos_sim
                # Average over patches and batch
                cos_sim = F.cosine_similarity(
                    feat_gt.detach(), feat_pred, dim=-1
                )  # (B, N)

                # Calculate per-sample loss (mean over patches)
                per_sample_loss = (1 - cos_sim).mean(dim=-1)  # (B,)

                # Apply low-sim gating if enabled
                if self.aux_dinov2_low_sim_gate:
                    # Only apply loss to samples with mean cos_sim < threshold
                    per_sample_cos_sim = cos_sim.mean(dim=-1)  # (B,)
                    low_sim_mask = (
                        per_sample_cos_sim < self.aux_dinov2_low_sim_thresh
                    ).float()
                    # Avoid division by zero when no samples pass the gate
                    mask_sum = low_sim_mask.sum()
                    if mask_sum > 0:
                        loss_aux_dinov2 = (
                            (per_sample_loss * low_sim_mask).sum()
                            / mask_sum
                            * self.aux_dinov2_loss_coeff
                        )
                    else:
                        loss_aux_dinov2 = torch.tensor(0.0, device=x.device)
                else:
                    loss_aux_dinov2 = (
                        per_sample_loss.mean() * self.aux_dinov2_loss_coeff
                    )

            ### 5d. Drifting V3 loss (Design B: gated attraction + repulsion from generated neighbors)
            loss_drifting_v3 = torch.tensor(0.0, device=x.device)
            drifting_v3_drift_norm = torch.tensor(0.0, device=x.device)
            if self.drifting_v3_loss:
                loss_drifting_v3, drifting_v3_info = self.compute_drifting_v3_loss(
                    x_gen=x_pixels_pred,
                    labels_gen=labels,
                    x_pos=x,
                    labels_pos=labels,
                )
                drifting_v3_drift_norm = torch.tensor(
                    drifting_v3_info["drifting_v3_drift_norm"], device=x.device
                )

            ### 6. Drift Distillation
            # This creates an improved target from the model's own prediction:
            # x_target = stop_grad(x_pred + η(t) * g), where g = -∇_{x_pred} E(x_pred)
            # Loss is ADDED to v-loss, not replacing it
            #
            # SOFT WEIGHTING: w(t) = t^power * (1-t)
            drift_cos_sim = torch.tensor(0.0, device=x.device)
            drift_weight_mean = torch.tensor(0.0, device=x.device)

            loss_dinov2_after_mask = torch.tensor(0.0, device=x.device)
            if self.dinov2_drop_zero_loss and dino_drop_mask is not None:
                if loss_dinov2.shape[0] > dino_drop_mask.sum():
                    loss_dinov2_after_mask = (
                        loss_dinov2.sum()
                        / loss_dinov2.shape[1]
                        / loss_dinov2.shape[2]
                        / (loss_dinov2.shape[0] - dino_drop_mask.sum())
                    )

            loss = (
                loss_pixels.mean(dim=(1, 2, 3)).mean() * self.pixels_loss_coef
                + loss_dinov2_after_mask * self.dinov2_loss_coef
                + loss_repa
                + loss_aux_dinov2
                + loss_drifting_v3 * self.drifting_v3_loss_coef
            )

            if loss.ndim == 0:
                result_dict = {
                    "total_loss": loss,
                    "loss_pixels": loss_pixels.mean() * self.pixels_loss_coef,
                    "loss_dinov2": loss_dinov2_after_mask * self.dinov2_loss_coef,
                    "pixel_snr": pixel_snr,
                    "dino_snr": dino_snr,
                    "snr_ratio": snr_ratio,
                }
                if self.enable_repa:
                    result_dict["loss_repa"] = loss_repa
                if self.aux_dinov2_loss:
                    result_dict["loss_aux_dinov2"] = loss_aux_dinov2
                if self.drifting_v3_loss:
                    result_dict["loss_drifting_v3"] = (
                        loss_drifting_v3 * self.drifting_v3_loss_coef
                    )
                    result_dict["drifting_v3_drift_norm"] = drifting_v3_drift_norm
                    result_dict["drifting_v3_attract_norm"] = torch.tensor(
                        drifting_v3_info["drifting_v3_attract_norm"], device=x.device
                    )
                    result_dict["drifting_v3_repel_norm"] = torch.tensor(
                        drifting_v3_info["drifting_v3_repel_norm"], device=x.device
                    )
                    result_dict["drifting_v3_gate_mean"] = torch.tensor(
                        drifting_v3_info["drifting_v3_gate_mean"], device=x.device
                    )
                return result_dict

            return loss

        else:
            e = torch.randn_like(x) * self.noise_scale
            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(self.t_eps)

            # Forward pass with REPA hidden states if enabled
            if self.enable_repa:
                x_pred = self.net(
                    z, t.flatten(), labels_dropped, return_repa_hidden_states=True
                )
                if isinstance(x_pred, tuple):
                    x_pred, repa_hidden_states = x_pred
                else:
                    repa_hidden_states = None
            else:
                x_pred = self.net(z, t.flatten(), labels_dropped)
                repa_hidden_states = None

            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

            # l2 loss
            loss_pixels = (v - v_pred) ** 2

            ### 5d. Drifting V3 loss (Design B: gated attraction + repulsion from generated neighbors)
            loss_drifting_v3 = torch.tensor(0.0, device=x.device)
            drifting_v3_drift_norm = torch.tensor(0.0, device=x.device)
            if self.drifting_v3_loss:
                loss_drifting_v3, drifting_v3_info = self.compute_drifting_v3_loss(
                    x_gen=x_pred,
                    labels_gen=labels,
                    x_pos=x,
                    labels_pos=labels,
                )
                drifting_v3_drift_norm = torch.tensor(
                    drifting_v3_info["drifting_v3_drift_norm"], device=x.device
                )

            ### Drift Distillation with DINOv2 (self-improvement style, works even when use_co_embed=False)
            # SOFT WEIGHTING: w(t) = t^power * (1-t)
            drift_cos_sim = torch.tensor(0.0, device=x.device)
            drift_weight_mean = torch.tensor(0.0, device=x.device)
            loss = loss_pixels.mean(dim=(1, 2, 3)).mean()

            # REPA loss (requires external encoder features passed as argument)
            loss_repa = torch.tensor(0.0, device=x.device)
            if (
                self.enable_repa
                and repa_hidden_states is not None
                and len(repa_hidden_states) > 0
            ):
                # External encoder features are passed as repa_target_features argument
                if repa_target_features is not None:
                    target_features = repa_target_features
                    # Use official iREPA pattern with projection losses
                    total_proj_loss = torch.tensor(0.0, device=x.device)
                    for proj_loss_name, proj_loss_fn, coeff in zip(
                        self.projection_loss_type, self.projection_loss, self.proj_coeff
                    ):
                        proj_loss = torch.tensor(0.0, device=x.device)
                        for layer_idx, hidden in repa_hidden_states.items():
                            # Project hidden states to target feature space
                            projected = self.net.repa_projectors[str(layer_idx)](hidden)
                            # Compute projection loss with external encoder features
                            proj_loss = proj_loss + proj_loss_fn(
                                target_features.detach(), projected
                            )
                        proj_loss = proj_loss / len(repa_hidden_states)
                        total_proj_loss = total_proj_loss + coeff * proj_loss
                    loss_repa = total_proj_loss

            # Auxiliary DINOv2 loss (works even when use_co_embed=False)
            loss_aux_dinov2 = torch.tensor(0.0, device=x.device)
            if self.aux_dinov2_loss:
                # Get the predicted clean image from the pixel branch
                # x_pred is the model's prediction of the clean image
                # x is the ground truth clean image

                # Resize and normalize images for DINOv2 (no gradient blocking for x_pred)
                x_gt_224 = self.resize_for_dinov2(x)
                x_gt_224_normalized = self.normalize_for_dinov2(x_gt_224)

                x_pred_224 = self.resize_for_dinov2(x_pred)
                x_pred_224_normalized = self.normalize_for_dinov2(x_pred_224)

                # Extract DINOv2 features (frozen model, but gradients flow through input)
                # Note: we DON'T use torch.no_grad() here so gradients can flow back
                aux_dinov2_out_gt = self.aux_dinov2(
                    x_gt_224_normalized, return_dict=True
                )
                aux_dinov2_out_pred = self.aux_dinov2(
                    x_pred_224_normalized, return_dict=True
                )

                # Get patch features (exclude CLS and register tokens)
                num_special_tokens = 1 + 4  # CLS + 4 registers
                feat_gt = aux_dinov2_out_gt.last_hidden_state[:, num_special_tokens:, :]
                feat_pred = aux_dinov2_out_pred.last_hidden_state[
                    :, num_special_tokens:, :
                ]

                # Compute cosine similarity loss: 1 - cos_sim
                # Average over patches and batch
                cos_sim = F.cosine_similarity(
                    feat_gt.detach(), feat_pred, dim=-1
                )  # (B, N)

                # Calculate per-sample loss (mean over patches)
                per_sample_loss = (1 - cos_sim).mean(dim=-1)  # (B,)

                # Apply low-sim gating if enabled
                if self.aux_dinov2_low_sim_gate:
                    # Only apply loss to samples with mean cos_sim < threshold
                    per_sample_cos_sim = cos_sim.mean(dim=-1)  # (B,)
                    low_sim_mask = (
                        per_sample_cos_sim < self.aux_dinov2_low_sim_thresh
                    ).float()
                    # Avoid division by zero when no samples pass the gate
                    mask_sum = low_sim_mask.sum()
                    if mask_sum > 0:
                        loss_aux_dinov2 = (
                            (per_sample_loss * low_sim_mask).sum()
                            / mask_sum
                            * self.aux_dinov2_loss_coeff
                        )
                    else:
                        loss_aux_dinov2 = torch.tensor(0.0, device=x.device)
                else:
                    loss_aux_dinov2 = (
                        per_sample_loss.mean() * self.aux_dinov2_loss_coeff
                    )

            total_loss = (
                loss * self.pixels_loss_coef
                + loss_repa
                + loss_aux_dinov2
                + loss_drifting_v3 * self.drifting_v3_loss_coef
            )

            if self.enable_repa or self.aux_dinov2_loss or self.drifting_v3_loss:
                result_dict = {
                    "total_loss": total_loss,
                    "loss_pixels": loss * self.pixels_loss_coef,
                }
                if self.enable_repa:
                    result_dict["loss_repa"] = loss_repa
                if self.aux_dinov2_loss:
                    result_dict["loss_aux_dinov2"] = loss_aux_dinov2
                if self.drifting_v3_loss:
                    result_dict["loss_drifting_v3"] = (
                        loss_drifting_v3 * self.drifting_v3_loss_coef
                    )
                    result_dict["drifting_v3_drift_norm"] = drifting_v3_drift_norm
                    result_dict["drifting_v3_attract_norm"] = torch.tensor(
                        drifting_v3_info["drifting_v3_attract_norm"], device=x.device
                    )
                    result_dict["drifting_v3_repel_norm"] = torch.tensor(
                        drifting_v3_info["drifting_v3_repel_norm"], device=x.device
                    )
                    result_dict["drifting_v3_gate_mean"] = torch.tensor(
                        drifting_v3_info["drifting_v3_gate_mean"], device=x.device
                    )
                return result_dict

            return total_loss

    @torch.no_grad()
    def generate(self, labels):
        """
        Generate images.
        Returns: (N, C, 256, 256) images in [-1, 1]
        """

        device = labels.device
        bsz = labels.size(0)

        if self.use_co_embed:
            z_pixels = self.noise_scale * torch.randn(
                bsz, 3, self.img_size, self.img_size, device=device
            )  # torch.Size([128, 3, 256, 256])

            z_dinov2 = self.noise_scale_dinov2 * torch.randn(
                bsz,
                (self.img_size // 16) * (self.img_size // 16),
                self.net.x_embedder.dinov2_dim,
                device=device,
            )  # torch.Size([128, 256, 768])

            timesteps = (
                torch.linspace(0.0, 1.0, self.steps + 1, device=device)
                .view(-1, *([1] * z_pixels.ndim))
                .expand(-1, bsz, -1, -1, -1)
            )
        else:
            z = self.noise_scale * torch.randn(
                bsz, 3, self.img_size, self.img_size, device=device
            )
            timesteps = (
                torch.linspace(0.0, 1.0, self.steps + 1, device=device)
                .view(-1, *([1] * z.ndim))
                .expand(-1, bsz, -1, -1, -1)
            )

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ODE integration
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            if self.use_co_embed:
                z_pixels, z_dinov2 = stepper(z_pixels, t, t_next, labels, zs=z_dinov2)
            else:
                z = stepper(z, t, t_next, labels)

        # Last step euler
        if self.use_co_embed:
            z_pixels, z_dinov2 = self._euler_step(
                z_pixels, timesteps[-2], timesteps[-1], labels, zs=z_dinov2
            )
            return z_pixels
        else:
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
            return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, zs=None):

        if self.use_co_embed:

            ### CFG interval setup
            low, high = self.cfg_interval
            interval_mask = (t < high) & ((low == 0) | (t > low))

            cfg_label = self.cfg_label  # cfg for label/text
            cfg_dino = self.cfg_dino  # cfg for dino stream

            ### Check if compositional CFG with DINOv2 condition is enabled
            if cfg_dino is not None:

                ### (1) cond (label=real, zs=real)
                x_pixels_cond, x_dinov2_cond = self.net(z, t.flatten(), labels, zs=zs)
                v_pixels_cond = (x_pixels_cond - z) / (1.0 - t).clamp_min(self.t_eps)
                v_dinov2_cond = (x_dinov2_cond - zs) / (1.0 - t.squeeze(-1)).clamp_min(
                    self.t_eps
                )

                ### (2) uncond (label=null, zs=real)
                if self.uncond_dino_null:
                    if self.dinov2_null_type in (
                        "attn_mask_asymmetric",
                        "attn_mask_symmetric",
                    ):
                        block_mask = torch.ones(
                            z.shape[0], dtype=torch.bool, device=z.device
                        )
                        x_pixels_uncond, x_dinov2_uncond = self.net(
                            z,
                            t.flatten(),
                            torch.full_like(labels, self.num_classes),
                            zs=zs,
                            block_dino_to_pixel=block_mask,
                            symmetric_attn_mask=(
                                self.dinov2_null_type == "attn_mask_symmetric"
                            ),
                        )
                    elif self.dinov2_null_type == "learned":
                        zs_null = self.dino_null_token.to(zs.device).expand(
                            zs.shape[0], zs.shape[1], -1
                        )
                        x_pixels_uncond, x_dinov2_uncond = self.net(
                            z,
                            t.flatten(),
                            torch.full_like(labels, self.num_classes),
                            zs=zs_null,
                        )
                    else:
                        zs_null = torch.zeros_like(zs)
                        x_pixels_uncond, x_dinov2_uncond = self.net(
                            z,
                            t.flatten(),
                            torch.full_like(labels, self.num_classes),
                            zs=zs_null,
                        )
                else:
                    x_pixels_uncond, x_dinov2_uncond = self.net(
                        z, t.flatten(), torch.full_like(labels, self.num_classes), zs=zs
                    )
                v_pixels_uncond = (x_pixels_uncond - z) / (1.0 - t).clamp_min(
                    self.t_eps
                )
                v_dinov2_uncond = (x_dinov2_uncond - zs) / (
                    1.0 - t.squeeze(-1)
                ).clamp_min(self.t_eps)

                cfg_scale_interval = torch.where(interval_mask, cfg_label, 1.0)
                cfg_dino_interval = torch.where(interval_mask, cfg_dino, 1.0)

                # Label-based CFG: v = v_uncond + cfg * (v_cond - v_uncond)
                v_pixels_pred = v_pixels_uncond + cfg_scale_interval * (
                    v_pixels_cond - v_pixels_uncond
                )
                v_dinov2_pred = v_dinov2_uncond + cfg_dino_interval.squeeze(-1) * (
                    v_dinov2_cond - v_dinov2_uncond
                )

            else:
                # Standard label-only CFG (original behavior)
                ### (1) cond (label=real, zs=real)
                x_pixels_cond, x_dinov2_cond = self.net(z, t.flatten(), labels, zs=zs)
                v_pixels_cond = (x_pixels_cond - z) / (1.0 - t).clamp_min(self.t_eps)
                v_dinov2_cond = (x_dinov2_cond - zs) / (1.0 - t.squeeze(-1)).clamp_min(
                    self.t_eps
                )

                ### (2) uncond on BOTH labels AND DINOv2 (label=null, zs=null)
                if self.dinov2_null_type in (
                    "attn_mask_asymmetric",
                    "attn_mask_symmetric",
                ):
                    block_mask = torch.ones(
                        z.shape[0], dtype=torch.bool, device=z.device
                    )
                    x_pixels_uncond, x_dinov2_uncond = self.net(
                        z,
                        t.flatten(),
                        torch.full_like(labels, self.num_classes),
                        zs=zs,
                        block_dino_to_pixel=block_mask,
                        symmetric_attn_mask=(
                            self.dinov2_null_type == "attn_mask_symmetric"
                        ),
                    )
                else:
                    if self.dinov2_null_type == "learned":
                        zs_null = self.dino_null_token.to(zs.device).expand(
                            zs.shape[0], zs.shape[1], -1
                        )
                    else:
                        zs_null = torch.zeros_like(zs)

                    x_pixels_uncond, x_dinov2_uncond = self.net(
                        z,
                        t.flatten(),
                        torch.full_like(labels, self.num_classes),
                        zs=zs_null,
                    )
                v_pixels_uncond = (x_pixels_uncond - z) / (1.0 - t).clamp_min(
                    self.t_eps
                )
                v_dinov2_uncond = (x_dinov2_uncond - zs) / (
                    1.0 - t.squeeze(-1)
                ).clamp_min(self.t_eps)

                cfg_scale_interval = torch.where(interval_mask, cfg_label, 1.0)

                # Label-based CFG: v = v_uncond + cfg * (v_cond - v_uncond)
                v_pixels_pred = v_pixels_uncond + cfg_scale_interval * (
                    v_pixels_cond - v_pixels_uncond
                )
                v_dinov2_pred = v_dinov2_uncond + cfg_scale_interval.squeeze(-1) * (
                    v_dinov2_cond - v_dinov2_uncond
                )

            return v_pixels_pred, v_dinov2_pred

        else:
            x_cond = self.net(z, t.flatten(), labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

            x_uncond = self.net(
                z, t.flatten(), torch.full_like(labels, self.num_classes)
            )
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

            # CFG interval
            low, high = self.cfg_interval
            interval_mask = (t < high) & ((low == 0) | (t > low))
            cfg_scale_interval = torch.where(interval_mask, self.cfg_label, 1.0)

            return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    def _shift_time(self, t, alpha):
        """Apply time shift: t' = alpha * t / (1 + (alpha - 1) * t)"""
        return alpha * t / (1 + (alpha - 1) * t)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, zs=None):
        if zs is not None:
            v_pixels_pred, v_dinov2_pred = self._forward_sample(z, t, labels, zs=zs)
            z_pixels_next = z + (t_next - t) * v_pixels_pred
            if self.use_dino_time_shift:
                alpha = self.dino_time_shift_alpha
                t_dino = self._shift_time(t, alpha)
                t_dino_next = self._shift_time(t_next, alpha)
                dt_dino = (t_dino_next - t_dino).squeeze(-1)
            else:
                dt_dino = (t_next - t).squeeze(-1)
            z_dinov2_next = zs + dt_dino * v_dinov2_pred
            return z_pixels_next, z_dinov2_next
        else:
            v_pred = self._forward_sample(z, t, labels)
            z_next = z + (t_next - t) * v_pred
            return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, zs=None):

        if zs is not None:

            v_pixels_pred_t, v_dinov2_pred_t = self._forward_sample(z, t, labels, zs=zs)

            if self.use_dino_time_shift:
                alpha = self.dino_time_shift_alpha
                t_dino = self._shift_time(t, alpha)
                t_dino_next = self._shift_time(t_next, alpha)
                dt_dino = (t_dino_next - t_dino).squeeze(-1)
            else:
                dt_dino = (t_next - t).squeeze(-1)

            z_pixels_next_euler = z + (t_next - t) * v_pixels_pred_t
            z_dinov2_next_euler = zs + dt_dino * v_dinov2_pred_t

            v_pixels_pred_t_next, v_dinov2_pred_t_next = self._forward_sample(
                z_pixels_next_euler, t_next, labels, zs=z_dinov2_next_euler
            )

            v_pixels_pred = 0.5 * (v_pixels_pred_t + v_pixels_pred_t_next)
            v_dinov2_pred = 0.5 * (v_dinov2_pred_t + v_dinov2_pred_t_next)

            z_pixels_next = z + (t_next - t) * v_pixels_pred
            z_dinov2_next = zs + dt_dino * v_dinov2_pred

            return z_pixels_next, z_dinov2_next

        else:
            v_pred_t = self._forward_sample(z, t, labels)

            z_next_euler = z + (t_next - t) * v_pred_t
            v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

            v_pred = 0.5 * (v_pred_t + v_pred_t_next)
            z_next = z + (t_next - t) * v_pred
            return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
