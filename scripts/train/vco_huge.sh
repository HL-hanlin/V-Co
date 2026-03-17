#!/bin/bash
#SBATCH --account=xxx                  # SLURM account
#SBATCH --qos=xxx                      # Quality of Service (QoS)
#SBATCH --partition=xxx                # Quality of Service (QoS)
#SBATCH --job-name=vco_huge            # Job name
#SBATCH --nodes=8                      # Number of nodes
#SBATCH --ntasks-per-node=1            # One task per node (torchrun handles GPU processes)
#SBATCH --gres=gpu:8                   # Number of GPUs per node
#SBATCH --time=7-00:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=/output/slurm-%j.out
#SBATCH --cpus-per-task=96


# =============================================================================
# Training
# =============================================================================

source /miniconda3/bin/activate
conda activate vco

export PATH="/miniconda3/envs/vco/bin:$PATH"
export OMP_NUM_THREADS=10

# Get master node address from SLURM (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Use SLURM_JOB_ID for dynamic port (same on all nodes in the same job)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

cd /path/to/VCo




srun torchrun --nproc_per_node=8 --nnodes=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    main_vco.py \
    --proj_dropout 0.0 \
    --P_mean -0.8 --P_std 0.8 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 16 --blr 5e-5 \
    --epochs 600 --warmup_epochs 5 \
    --gen_bsz 16 --num_images 50000 --interval_min 0.1 --interval_max 1.0 --eval_freq 20 \
    --auto_resume \
    --online_eval \
    --use_wandb \
    --model JiT-H/16-co \
    --output_dir '/path/to/output_dir' \
    --wandb_entity 'your_entity' \
    --wandb_run_name 'vco_huge' \
    --data_path '/path/to/imagenet/' \
    --num_workers 12 \
    --use_co_embed \
    --use_dinov2 \
    --use_dino_from_rae \
    --dinov2_loss_coef 0.1 \
    --use_gated_co_embed \
    --noise_scale_dinov2 1.0 \
    --jit_refiner_layers 0 \
    --use_mmdit \
    --separate_qkv \
    --use_conv2d_dino_proj \
    --label_drop_prob 0.0 \
    --dinov2_drop_prob 0.0 \
    --label_dinov2_drop_prob 0.1 \
    --uncond_dino_null \
    --dinov2_null_type 'attn_mask_asymmetric' \
    --dinov2_drop_zero_loss \
    --cfg 1.5 \
    --cfg_dino 1.5 \
    --class_balanced_sampling \
    --num_classes_per_batch 1 \
    --num_samples_per_class 16 \
    --drifting_v3_loss \
    --drifting_v3_loss_coef 10.0 \
    --drifting_v3_feat_type 'cls' \
    --drifting_v3_gate_tau 10 \
    --drifting_v3_repulsion_tau 0.2
