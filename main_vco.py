import argparse
import copy
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import util.misc as misc
from denoiser import Denoiser
from engine_vco import evaluate, evaluate_cfg_sweep, train_one_epoch
from tabulate import tabulate
from util.crop import center_crop_arr


def get_args_parser():

    parser = argparse.ArgumentParser("JiT", add_help=False)

    # architecture
    parser.add_argument(
        "--model",
        default="JiT-B/16",
        type=str,
        metavar="MODEL",
        help="Name of the model to train",
    )
    parser.add_argument("--img_size", default=256, type=int, help="Image size")
    parser.add_argument(
        "--attn_dropout", type=float, default=0.0, help="Attention dropout rate"
    )
    parser.add_argument(
        "--proj_dropout", type=float, default=0.0, help="Projection dropout rate"
    )

    # DINOv2 encoder options
    parser.add_argument(
        "--use_dinov2",
        action="store_true",
        help="Use DINOv2 as patch embedder instead of BottleneckPatchEmbed",
    )
    parser.add_argument(
        "--use_dino_from_rae",
        action="store_true",
        help="whether to use the dinov2 from rae",
    )
    parser.add_argument(
        "--dinov2_model_name",
        default="facebook/dinov2-with-registers-base",
        type=str,
        help="DINOv2 model name or path",
    )
    parser.add_argument(
        "--freeze_dinov2",
        action="store_true",
        help="Freeze DINOv2 weights (default behavior if neither flag is set)",
    )
    parser.add_argument(
        "--unfreeze_dinov2", action="store_true", help="Make DINOv2 weights trainable"
    )
    # training
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="Epochs to warm up LR"
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size per GPU (effective batch size = batch_size * # GPUs)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, metavar="LR", help="Learning rate (absolute)"
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=5e-5,
        metavar="LR",
        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="Minimum LR for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--lr_schedule", type=str, default="constant", help="Learning rate schedule"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay (default: 0.0)"
    )
    parser.add_argument(
        "--ema_decay1",
        type=float,
        default=0.9999,
        help="The first ema to track. Use the first ema for sampling by default.",
    )
    parser.add_argument(
        "--ema_decay2", type=float, default=0.9996, help="The second ema to track"
    )
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument(
        "--noise_scale", default=1.0, type=float, help="Noise scale for pixel space"
    )
    parser.add_argument(
        "--noise_scale_dinov2",
        default=None,
        type=float,
        help="Noise scale for DINOv2 features. If None, uses same as --noise_scale",
    )
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument(
        "--label_dinov2_drop_prob",
        default=0.0,
        type=float,
        help="Probability of jointly dropping BOTH label and DINOv2 condition. "
        "When > 0, a single Bernoulli draw determines whether both are dropped together. "
        "Mutually exclusive with independent --label_drop_prob / --dinov2_drop_prob.",
    )
    parser.add_argument(
        "--dinov2_drop_prob",
        default=0.0,
        type=float,
        help="Probability of dropping DINOv2 condition during training for CFG. "
        "Set to 0.1 to enable CFG for DINOv2.",
    )
    parser.add_argument(
        "--dinov2_drop_zero_loss",
        action="store_true",
        default=False,
        help="When True, samples where DINOv2 was dropped (via dinov2_drop_prob or "
        "label_dinov2_drop_prob) contribute zero DINOv2 loss to the total loss.",
    )
    parser.add_argument(
        "--dinov2_null_type",
        default="zero",
        type=str,
        choices=[
            "zero",
            "mean",
            "noise",
            "learned",
            "attn_mask_asymmetric",
            "attn_mask_symmetric",
        ],
        help="Type of null condition for DINOv2 CFG when dropping: "
        "'zero' = use zeros (appropriate when standardize_dinov2=True), "
        "'learned' = use a learnable null DINOv2 token (nn.Parameter), "
        "'mean' = use pre-computed mean (requires standardize_dinov2 stats), "
        "'noise' = use pure noise at t=0 (noise_scale_dinov2 * randn), "
        "'attn_mask_asymmetric' = block pixel_q->DINO_k attention only (DINO can still see pixel), "
        "'attn_mask_symmetric' = block both pixel_q->DINO_k and DINO_q->pixel_k (full isolation).",
    )

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="Starting epoch"
    )
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for faster GPU transfers",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # class-balanced sampling (alternative to IID)
    parser.add_argument(
        "--class_balanced_sampling",
        action="store_true",
        default=False,
        help="Use class-balanced batch sampler instead of IID. "
        "Each mini-batch will contain exactly N classes × M samples per class.",
    )
    parser.add_argument(
        "--num_classes_per_batch",
        default=1,
        type=int,
        help="N: number of distinct classes per mini-batch.",
    )
    parser.add_argument(
        "--num_samples_per_class",
        default=32,
        type=int,
        help="M: number of samples per class per mini-batch. "
        "Effective batch size = num_classes_per_batch × num_samples_per_class.",
    )

    # sampling
    parser.add_argument(
        "--sampling_method", default="heun", type=str, help="ODE samping method"
    )
    parser.add_argument(
        "--num_sampling_steps", default=50, type=int, help="Sampling steps"
    )
    parser.add_argument(
        "--cfg", default=1.0, type=float, help="Classifier-free guidance factor"
    )
    parser.add_argument(
        "--cfg_dino",
        default=None,
        type=float,
        help="Classifier-free guidance factor for DINOv2 features (label-based CFG). If None, uses the same as --cfg.",
    )
    parser.add_argument(
        "--cfg_dinov2_cond",
        default=None,
        type=float,
        help="CFG scale for DINOv2 condition guidance (zs-based CFG). "
        "If set, adds an extra forward pass with zs=zeros for DINOv2 unconditional. "
        "Requires training with --dinov2_drop_prob > 0. If None, disabled.",
    )
    parser.add_argument(
        "--interval_min", default=0.1, type=float, help="CFG interval min"
    )
    parser.add_argument(
        "--interval_max", default=1.0, type=float, help="CFG interval max"
    )
    parser.add_argument(
        "--num_images", default=50000, type=int, help="Number of images to generate"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=20, help="Frequency (in epochs) for evaluation"
    )
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--evaluate_gen", action="store_true")
    parser.add_argument(
        "--cfg_sweep",
        default=None,
        type=str,
        help="Comma-separated list of CFG values for grid search evaluation. "
        "Both --cfg and --cfg_dino are set to each value. "
        "Example: '1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0'",
    )
    parser.add_argument(
        "--gen_bsz", type=int, default=256, help="Generation batch size"
    )
    parser.add_argument(
        "--fid_statistics_file",
        default="fid_stats/jit_in256_stats.npz",
        type=str,
        help="Path to FID statistics file (.npz). If None, uses default path based on img_size",
    )

    # dataset
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument("--class_num", default=1000, type=int)

    # checkpointing
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Directory to save outputs (empty for no saving)",
    )
    parser.add_argument(
        "--resume", default="", help="Folder that contains checkpoint to resume from"
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from latest checkpoint in output_dir if available",
    )
    parser.add_argument(
        "--save_last_freq",
        type=int,
        default=5,
        help="Frequency (in epochs) to save checkpoints",
    )
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training/testing"
    )

    # distributed training
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )

    # wandb logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb_project", default="JiT", type=str, help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity",
        default="xxx",
        type=str,
        help="Wandb entity (team or username)",
    )
    parser.add_argument(
        "--wandb_run_name", default=None, type=str, help="Wandb run name"
    )
    parser.add_argument(
        "--uncond_dino_null", action="store_true", help="uncond_dino_null"
    )
    # jit-cot from latent forcing
    parser.add_argument(
        "--use_jit_cot_model", action="store_true", help="uncond_dino_null"
    )
    # co-denoising
    parser.add_argument(
        "--use_co_embed", action="store_true", help="Enable co-denoising"
    )
    parser.add_argument(
        "--use_gated_co_embed", action="store_true", help="Enable co-denoising"
    )
    parser.add_argument(
        "--dinov2_loss_coef", default=1.0, type=float, help="loss weighting for dinov2"
    )
    parser.add_argument(
        "--pixels_loss_coef", default=1.0, type=float, help="loss weighting for pixels"
    )
    parser.add_argument(
        "--use_mmdit", action="store_true", help="Enable MMDiT architecture"
    )
    parser.add_argument(
        "--separate_qkv",
        action="store_true",
        help="Use separate QKV projections for pixel and DINOv2 in MMDiTBlock (better for bridging different feature spaces)",
    )
    parser.add_argument(
        "--jit_dino_proj",
        action="store_true",
        help="Use JIT style refiner (JiTBlock with RoPE + AdaLN) for DINOv2 input projection.",
    )
    parser.add_argument(
        "--use_conv2d_dino_proj",
        action="store_true",
        help="Use Conv2d (1x1) instead of Linear MLP for DINOv2 input projection.",
    )
    parser.add_argument(
        "--jit_pixel_proj",
        action="store_true",
        help="Use JIT style refiner (JiTBlock with RoPE + AdaLN) for pixel input projection.",
    )
    parser.add_argument(
        "--jit_refiner_layers",
        default=0,
        type=int,
        help="Number of transformer layers in JIT style refiners (default: 2).",
    )
    parser.add_argument(
        "--jit_dino_head",
        action="store_true",
        help="Use JIT style refiner (JiTBlock with RoPE + AdaLN) for DINOv2 output head.",
    )
    parser.add_argument(
        "--jit_pixel_head",
        action="store_true",
        help="Use JIT style refiner (JiTBlock with RoPE + AdaLN) for pixel output head.",
    )

    # Normalization options for co-denoising (similar to RAE repo)
    parser.add_argument(
        "--normalize_dinov2_latent",
        action="store_true",
        help="Normalize DINOv2 features using pre-computed mean/var statistics. "
        "Requires --dinov2_latent_stats_path to be set.",
    )
    parser.add_argument(
        "--dinov2_latent_stats_path",
        default=None,
        type=str,
        help="Path to pre-computed DINOv2 latent statistics (.pt file with 'mean' and 'var' keys). "
        "Required when --normalize_dinov2_latent is set.",
    )
    # Shared JiT architecture (alternative to MMDiT dual-stream)
    parser.add_argument(
        "--use_shared_jit",
        action="store_true",
        help="Use shared JiT architecture: K separate pre-blocks for each stream, M shared middle blocks, "
        "K separate post-blocks. K = jit_refiner_layers, M = depth - 2*K. "
        "During inference, only pixel path is used (DINOv2 blocks discarded).",
    )

    parser.add_argument(
        "--use_channel_concat",
        action="store_true",
        help="When using SharedJiT, token-concatenate pixel and DINOv2 features along sequence dim "
        "before shared middle blocks, enabling cross-modal attention. Features are split back "
        "after shared blocks for independent post-processing.",
    )

    parser.add_argument(
        "--use_token_concat",
        action="store_true",
        help="When using JiTCoT, concatenate pixel and DINOv2 tokens along the sequence dimension "
        "(B, N, H) + (B, N, H) -> (B, 2N, H) with shared positional embeddings, then split "
        "back after the transformer for separate final layers.",
    )

    parser.add_argument(
        "--match_pixel_norm",
        type=float,
        default=0.485,
        help="Scaling factor for RAE normalized DINOv2 features. "
        "Passed to RAE(match_pixel_norm=...) to control the output magnitude.",
    )
    parser.add_argument(
        "--use_dino_time_shift",
        action="store_true",
        help="Instead of scaling DINOv2 features, shift the DINOv2 noise schedule "
        "so that SNR_dino(t') = SNR_scaled_dino(t). Equivalent to feature rescaling.",
    )
    parser.add_argument(
        "--dino_time_shift_alpha",
        type=float,
        default=0.485,
        help="Time shift factor alpha. t' = alpha*t / (1 + (alpha-1)*t). "
        "Should match the RMS ratio RMS_pixels / RMS_dino.",
    )

    parser.add_argument(
        "--use_direct_addition",
        action="store_true",
        help="When using SharedJiT, element-wise add pixel and DINOv2 features before shared "
        "middle blocks. No pre-blocks, only shared + post-blocks.",
    )

    # Auxiliary DINOv2 loss for pixel branch (scalar penalty)
    parser.add_argument(
        "--aux_dinov2_loss",
        action="store_true",
        help="Enable auxiliary DINOv2 loss: 1 - cosine_similarity(DINOv2(x_gt), DINOv2(x_pred)). "
        "Uses a frozen DINOv2 model to compare ground truth and predicted images.",
    )
    parser.add_argument(
        "--aux_dinov2_loss_coeff",
        default=1.0,
        type=float,
        help="Coefficient for auxiliary DINOv2 loss. Default: 1.0",
    )
    parser.add_argument(
        "--aux_dinov2_low_sim_gate",
        action="store_true",
        help="Only apply aux_dinov2_loss to samples with cos_sim < threshold. "
        "This focuses the DINO loss on samples that need more semantic guidance.",
    )
    parser.add_argument(
        "--aux_dinov2_low_sim_thresh",
        default=0.5,
        type=float,
        help="Threshold for low-sim gating. Only samples with cos_sim < this value "
        "will contribute to aux_dinov2_loss. Default: 0.5",
    )
    parser.add_argument(
        "--dh_depth",
        default=2,
        type=int,
        help="for model CoT, the depth of DH blocks",
    )
    # Drifting V3 loss (per-sample positive, batch-wide negatives)
    parser.add_argument(
        "--drifting_v3_loss",
        action="store_true",
        help="Enable drifting V3 loss (Design B: gated attraction + repulsion from generated neighbors). "
        "For each sample i, attracts feat_gen[i] toward feat_pos[i] (paired GT) and repels from "
        "other same-class generated features. Gate s_i controls the balance based on proximity to target.",
    )
    parser.add_argument(
        "--drifting_v3_loss_coef",
        default=1.0,
        type=float,
        help="Coefficient for drifting V3 loss. Default: 1.0",
    )
    parser.add_argument(
        "--drifting_v3_feat_type",
        default="cls",
        type=str,
        choices=["cls", "avg_pool2d"],
        help="Feature type for drifting V3 loss. Default: 'cls'",
    )
    parser.add_argument(
        "--drifting_v3_gate_tau",
        default=1.0,
        type=float,
        help="Temperature for the similarity gate s_i = exp(-||z - y_i||^2 / tau). "
        "Smaller tau = gate turns off repulsion earlier. Default: 1.0",
    )
    parser.add_argument(
        "--use_fixed_gate_tau",
        action="store_true",
        default=False,
        help="If set, use s_i = drifting_v3_gate_tau as a fixed constant gate "
        "instead of s_i = exp(-dist / tau). Default: False",
    )
    parser.add_argument(
        "--drifting_v3_repulsion_tau",
        default=1.0,
        type=float,
        help="Temperature for the repulsion kernel exp(-||z - z_k||^2 / tau). "
        "Controls how strongly nearby generated samples repel. Default: 1.0",
    )
    parser.add_argument(
        "--drifting_v3_w_attract",
        default=1.0,
        type=float,
        help="Weight for the attraction term w+(t). Default: 1.0",
    )
    parser.add_argument(
        "--drifting_v3_w_repel",
        default=1.0,
        type=float,
        help="Weight for the repulsion term w-(t). Default: 1.0",
    )

    # REPA (Representation Alignment) arguments - matching official iREPA implementation
    parser.add_argument(
        "--enable_repa",
        action="store_true",
        help="Enable REPA (Representation Alignment) loss for aligning diffusion model hidden states with external vision encoder features.",
    )
    parser.add_argument(
        "--encoder_depth",
        type=int,
        default=4,
        help="Depth (layer index) at which to extract hidden states for REPA alignment. Default: 8",
    )
    parser.add_argument(
        "--projector_dim",
        type=int,
        default=2048,
        help="Hidden dimension of the REPA projector MLP. Default: 2048",
    )
    parser.add_argument(
        "--enc_type",
        type=str,
        default="dinov2-vit-b",
        help="External vision encoder type for REPA targets. Format: encoder_type-architecture-model_config. "
        "Examples: 'dinov2-vit-b', 'dinov2reg-vit-b', 'siglip2-vit-b'. Default: 'dinov2-vit-b'",
    )
    parser.add_argument(
        "--proj_coeff",
        type=str,
        default="0.5",
        help="Comma-separated coefficients for REPA projection loss. Default: '0.5'",
    )
    parser.add_argument(
        "--projection_layer_type",
        type=str,
        default="mlp",
        choices=["mlp", "linear", "conv"],
        help="Type of projection layer: 'mlp' (3-layer with SiLU), 'linear', or 'conv'. Default: 'mlp'",
    )
    parser.add_argument(
        "--proj_kwargs_kernel_size",
        type=int,
        default=3,
        choices=[1, 3, 5, 7],
        help="Kernel size for conv projection layer. Default: 3",
    )
    parser.add_argument(
        "--projection_loss_type",
        type=str,
        default="cosine",
        help="Comma-separated list of projection loss types. Default: 'cosine'",
    )
    parser.add_argument(
        "--spnorm_method",
        type=str,
        default="none",
        choices=["none", "zscore"],
        help="Spatial normalization method for vision encoder features. Default: 'none'",
    )
    parser.add_argument(
        "--zscore_alpha",
        type=float,
        default=0.8,
        help="Alpha parameter for z-score spatial normalization. Default: 0.8",
    )

    return parser


def count_params(model, prefix=None, contains=None):
    total_trainable_params_count = 0
    total_params_count = 0
    for n, p in model.named_parameters():
        if contains is None:
            if p.requires_grad:
                total_trainable_params_count += p.numel()
        elif contains in n:
            if p.requires_grad:
                total_trainable_params_count += p.numel()
        total_params_count += p.numel()
    print(
        f"Total params in {prefix}: {total_params_count * 1e-6} M | trainable: {total_trainable_params_count * 1e-6} M"
    )


def main(args):

    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # Initialize wandb (only on main process)
    if args.use_wandb and misc.get_rank() == 0:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir,
        )

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ]
    )

    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    print(dataset_train)

    if args.class_balanced_sampling:
        from sample_queue import DistributedClassBalancedBatchSampler

        sampler_train = DistributedClassBalancedBatchSampler(
            dataset_train,
            num_classes_per_batch=args.num_classes_per_batch,
            num_samples_per_class=args.num_samples_per_class,
            num_replicas=num_tasks,
            rank=global_rank,
            seed=args.seed,
        )
        eff_batch_size = (
            args.num_classes_per_batch * args.num_samples_per_class * num_tasks
        )
        print(
            f"Class-balanced sampling: {args.num_classes_per_batch} classes × "
            f"{args.num_samples_per_class} samples/class = "
            f"{args.num_classes_per_batch * args.num_samples_per_class} per GPU, "
            f"eff_batch_size={eff_batch_size}"
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_sampler=sampler_train,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Prepare REPA kwargs if enabled
    repa_kwargs = None
    if getattr(args, "enable_repa", False):
        from spnorm import SpatialNormalization
        from vision_encoder import load_encoders

        # Load external vision encoders for REPA
        encoders = load_encoders(args.enc_type, device, args.img_size)
        spnorm = SpatialNormalization(args.spnorm_method)
        repa_kwargs = {
            "encoders": encoders,
            "spnorm": spnorm,
            "zscore_alpha": args.zscore_alpha,
        }
        print(f"REPA enabled with encoder: {args.enc_type}")
        print(f"REPA kwargs: {repa_kwargs}")

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided or auto-resume from output_dir
    checkpoint_path = None
    if args.resume:
        if ".pth" in args.resume:
            checkpoint_path = args.resume
        else:
            checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth")
    elif args.auto_resume and args.output_dir:
        # Auto-resume: look for checkpoint-last.pth in output_dir
        auto_resume_path = os.path.join(args.output_dir, "checkpoint-last.pth")
        if os.path.exists(auto_resume_path):
            checkpoint_path = auto_resume_path
            print(f"Auto-resuming from {checkpoint_path}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])

        ema_state_dict1 = checkpoint["model_ema1"]
        ema_state_dict2 = checkpoint["model_ema2"]
        model_without_ddp.ema_params1 = [
            ema_state_dict1[name].cuda()
            for name, _ in model_without_ddp.named_parameters()
        ]
        model_without_ddp.ema_params2 = [
            ema_state_dict2[name].cuda()
            for name, _ in model_without_ddp.named_parameters()
        ]
        print(f"Resumed checkpoint from {checkpoint_path}")

        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(
            list(model_without_ddp.parameters())
        )
        model_without_ddp.ema_params2 = copy.deepcopy(
            list(model_without_ddp.parameters())
        )
        print("Training from scratch")

    if misc.get_rank() == 0:
        stat = []
        for i, (n, p) in enumerate(model.named_parameters()):
            if not p.requires_grad:
                stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
        stat = []
        for i, (n, p) in enumerate(model.named_parameters()):
            if p.requires_grad:
                stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

    count_params(model, prefix="model")
    if not args.use_jit_cot_model:
        count_params(model.module.net.x_embedder, prefix="x_embedder")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                if args.cfg_sweep is not None:
                    evaluate_cfg_sweep(
                        model_without_ddp,
                        args,
                        0,
                        batch_size=args.gen_bsz,
                        log_writer=log_writer,
                    )
                else:
                    evaluate(
                        model_without_ddp,
                        args,
                        0,
                        batch_size=args.gen_bsz,
                        log_writer=log_writer,
                    )
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            if args.class_balanced_sampling:
                data_loader_train.batch_sampler.set_epoch(epoch)
            else:
                data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            model_without_ddp,
            data_loader_train,
            optimizer,
            device,
            epoch,
            repa_kwargs=repa_kwargs,
            log_writer=log_writer,
            args=args,
        )

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
            )

        # Save milestone checkpoints that are never overwritten
        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (
            epoch % args.eval_freq == 0 or epoch + 1 == args.epochs
        ):
            # torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(
                    model_without_ddp,
                    args,
                    epoch,
                    batch_size=args.gen_bsz,
                    log_writer=log_writer,
                )
            # torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time:", total_time_str)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
