"""
Vision encoders for REPA (Representation Alignment).
Simplified from official iREPA implementation.
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from torchvision.transforms import Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class VisionEncoder(ABC):
    """Base class for all vision encoders"""

    def __init__(self, encoder_type: str, architecture: str, model_config: str,
                 device: torch.device, resolution: int = 256, accelerator=None):
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim = None
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load and initialize the encoder model"""
        pass

    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw images
        Args:
            x: Raw images tensor (B, C, H, W) in range [0, 255]
        Returns:
            Preprocessed tensor ready for encoder
        """
        pass

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through encoder
        Args:
            x: Preprocessed images
        Returns:
            Dictionary with:
                - 'x_norm_clstoken': (B, D) CLS token or None if not available
                - 'x_norm_patchtokens': (B, T, D) patch tokens
        """
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        else:
            return {
                'x_norm_clstoken': None,
                'x_norm_patchtokens': out
            }

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, device):
        """Move model to device"""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class DINOv2Encoder(VisionEncoder):
    """DINOv2 encoder implementation"""

    def load_model(self):
        # Determine if using register tokens
        use_reg = 'reg' in self.encoder_type

        # Load model from torch hub
        model_name = f'dinov2_vit{self.model_config}14{"_reg" if use_reg else ""}'

        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()

        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )

        # Set embed dim
        self._embed_dim = self.model.embed_dim

        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINOv2 returns a dictionary with cls and patch tokens
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }



# Registry mapping encoder types to classes
ENCODER_REGISTRY = {
    'dinov2': DINOv2Encoder,
    'dinov2reg': DINOv2Encoder,
}


def create_encoder(encoder_string: str, device: torch.device,
                   resolution: int = 256, accelerator=None) -> VisionEncoder:
    """
    Factory function to create encoder from string specification

    Args:
        encoder_string: Format "encoder_type-architecture-model_config"
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training

    Returns:
        VisionEncoder instance
    """
    parts = encoder_string.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid encoder string format: {encoder_string}. "
                        f"Expected format: encoder_type-architecture-model_config")

    encoder_type, architecture, model_config = parts

    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available types: {list(ENCODER_REGISTRY.keys())}")

    encoder_class = ENCODER_REGISTRY[encoder_type]
    encoder = encoder_class(encoder_type, architecture, model_config,
                            device, resolution, accelerator)
    encoder.load_model()

    return encoder


@torch.no_grad()
def load_encoders(enc_type: str, device: torch.device, resolution: int = 256,
                  accelerator=None) -> List[VisionEncoder]:
    """
    Load multiple encoders from comma-separated string

    Args:
        enc_type: Comma-separated encoder specifications (e.g., "dinov2-vit-b")
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training

    Returns:
        List of VisionEncoder instances
    """
    enc_names = enc_type.split(',')
    encoders = []

    for enc_name in enc_names:
        # Parse encoder specification
        parts = enc_name.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid encoder format: {enc_name}")

        encoder = create_encoder(enc_name, device, resolution, accelerator)
        encoder.eval()
        encoders.append(encoder)

    return encoders
