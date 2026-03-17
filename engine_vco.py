import copy
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch_fidelity
import util.lr_sched as lr_sched
import util.misc as misc

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_one_epoch(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    repa_kwargs=None,
    log_writer=None,
    args=None,
):

    # REPA kwargs
    encoders = repa_kwargs.get("encoders", []) if repa_kwargs else []
    spnorm = repa_kwargs.get("spnorm", None) if repa_kwargs else None
    zscore_alpha = repa_kwargs.get("zscore_alpha", 0.8) if repa_kwargs else 0.8

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )

        # Extract REPA target features if encoders are provided
        zs_repa = None
        if encoders:
            with torch.no_grad():
                zs_repa = []
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for encoder in encoders:
                        # Preprocess the image using encoder's built-in method
                        x_ = x.to(device, non_blocking=True)
                        raw_image_ = encoder.preprocess(x_)

                        # Encode the features
                        features = encoder.forward_features(raw_image_)

                        # Apply spatial normalization
                        if spnorm is not None:
                            z = spnorm(
                                features["x_norm_patchtokens"],
                                zscore_alpha=zscore_alpha,
                            )
                        else:
                            z = features["x_norm_patchtokens"]

                        zs_repa.append(z)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        # Prepare REPA target features for forward pass
        repa_target_features = None
        if zs_repa:
            # For now, use the first encoder's features
            repa_target_features = zs_repa[0] if len(zs_repa) == 1 else zs_repa

        loss_pixels, loss_dinov2 = None, None
        pixel_snr, dino_snr, snr_ratio = None, None, None
        loss_repa = None
        loss_aux_dinov2 = None
        loss_drifting_v3 = None
        drifting_v3_drift_norm = None
        drifting_v3_attract_norm = None
        drifting_v3_repel_norm = None
        drifting_v3_gate_mean = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, labels, repa_target_features=repa_target_features)
            if type(loss) == dict:
                loss_pixels = loss.get("loss_pixels")
                if loss_pixels is not None:
                    loss_pixels = loss_pixels.item()
                loss_dinov2 = loss.get("loss_dinov2")
                if loss_dinov2 is not None:
                    loss_dinov2 = loss_dinov2.item()
                pixel_snr = loss.get("pixel_snr", None)
                dino_snr = loss.get("dino_snr", None)
                snr_ratio = loss.get("snr_ratio", None)
                loss_repa_tensor = loss.get("loss_repa", None)
                loss_aux_dinov2_tensor = loss.get("loss_aux_dinov2", None)
                if pixel_snr is not None:
                    pixel_snr = pixel_snr.item()
                    dino_snr = dino_snr.item()
                    snr_ratio = snr_ratio.item()
                if loss_repa_tensor is not None:
                    loss_repa = loss_repa_tensor.item()
                if loss_aux_dinov2_tensor is not None:
                    loss_aux_dinov2 = loss_aux_dinov2_tensor.item()
                loss_drifting_v3_tensor = loss.get("loss_drifting_v3", None)
                drifting_v3_drift_norm_tensor = loss.get("drifting_v3_drift_norm", None)
                if loss_drifting_v3_tensor is not None:
                    loss_drifting_v3 = loss_drifting_v3_tensor.item()
                if drifting_v3_drift_norm_tensor is not None:
                    drifting_v3_drift_norm = drifting_v3_drift_norm_tensor.item()
                drifting_v3_attract_norm_tensor = loss.get(
                    "drifting_v3_attract_norm", None
                )
                drifting_v3_repel_norm_tensor = loss.get("drifting_v3_repel_norm", None)
                drifting_v3_gate_mean_tensor = loss.get("drifting_v3_gate_mean", None)
                if drifting_v3_attract_norm_tensor is not None:
                    drifting_v3_attract_norm = drifting_v3_attract_norm_tensor.item()
                if drifting_v3_repel_norm_tensor is not None:
                    drifting_v3_repel_norm = drifting_v3_repel_norm_tensor.item()
                if drifting_v3_gate_mean_tensor is not None:
                    drifting_v3_gate_mean = drifting_v3_gate_mean_tensor.item()
                loss = loss["total_loss"]

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norms for monitoring (before optimizer.step() clears them)
        pixel_grad_norm = 0.0
        dinov2_grad_norm = 0.0
        total_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_grad_norm**2

                # Classify gradient by parameter name
                if "dino" in name.lower():
                    dinov2_grad_norm += param_grad_norm**2
                elif "pix" in name.lower():
                    pixel_grad_norm += param_grad_norm**2

        # Take square root to get L2 norm
        total_grad_norm = total_grad_norm**0.5
        pixel_grad_norm = pixel_grad_norm**0.5
        dinov2_grad_norm = dinov2_grad_norm**0.5

        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)
                log_writer.add_scalar("total_grad_norm", total_grad_norm, epoch_1000x)
                log_writer.add_scalar("pixel_grad_norm", pixel_grad_norm, epoch_1000x)
                log_writer.add_scalar("dinov2_grad_norm", dinov2_grad_norm, epoch_1000x)

        # wandb logging
        if WANDB_AVAILABLE and args.use_wandb and misc.get_rank() == 0:
            if data_iter_step % args.log_freq == 0:
                global_step = epoch * len(data_loader) + data_iter_step
                wandb.log(
                    {
                        "train/loss": loss_value_reduce,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/total_grad_norm": total_grad_norm,
                        "train/pixel_grad_norm": pixel_grad_norm,
                        "train/dinov2_grad_norm": dinov2_grad_norm,
                        "train/grad_norm_ratio": pixel_grad_norm
                        / (dinov2_grad_norm + 1e-8),
                    },
                    step=global_step,
                )
                if loss_pixels is not None:
                    log_dict = {
                        "train/loss_pixels": loss_pixels,
                        "train/loss_dinov2": loss_dinov2,
                    }
                    if snr_ratio is not None:
                        log_dict.update(
                            {
                                "train/pixel_snr": pixel_snr,
                                "train/dino_snr": dino_snr,
                                "train/snr_ratio": snr_ratio,
                            }
                        )
                    if loss_repa is not None:
                        log_dict["train/loss_repa"] = loss_repa
                    if loss_aux_dinov2 is not None:
                        log_dict["train/loss_aux_dinov2"] = loss_aux_dinov2
                    if loss_drifting_v3 is not None:
                        log_dict["train/loss_drifting_v3"] = loss_drifting_v3
                    if drifting_v3_drift_norm is not None:
                        log_dict["train/drifting_v3_drift_norm"] = (
                            drifting_v3_drift_norm
                        )
                    if drifting_v3_attract_norm is not None:
                        log_dict["train/drifting_v3_attract_norm"] = (
                            drifting_v3_attract_norm
                        )
                    if drifting_v3_repel_norm is not None:
                        log_dict["train/drifting_v3_repel_norm"] = (
                            drifting_v3_repel_norm
                        )
                    if drifting_v3_gate_mean is not None:
                        log_dict["train/drifting_v3_gate_mean"] = drifting_v3_gate_mean
                    wandb.log(log_dict, step=global_step)

    # Log epoch-level statistics
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if WANDB_AVAILABLE and args.use_wandb and misc.get_rank() == 0:
        wandb.log(
            {
                "epoch/train_loss_avg": metric_logger.meters["loss"].global_avg,
                "epoch/lr": metric_logger.meters["lr"].value,
                "epoch": epoch,
            }
        )


def _evaluate_single_cfg(
    model_without_ddp, args, epoch, batch_size, log_writer, cfg_label, cfg_dino
):
    """
    Run generation and compute FID/IS for a single cfg_scale value.
    The model's cfg_scale is temporarily overridden.
    Assumes EMA weights are already loaded and will NOT be restored here.
    """

    original_cfg_scale = model_without_ddp.cfg_label
    model_without_ddp.cfg_label = cfg_label

    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-cfgdino{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method,
            model_without_ddp.steps,
            cfg_label,
            cfg_dino,
            model_without_ddp.cfg_interval[0],
            model_without_ddp.cfg_interval[1],
            args.num_images,
            args.img_size,
        ),
    )
    if args.uncond_dino_null:
        print("##### using dinov2 null tokens as uncond #####")
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    class_num = args.class_num
    assert (
        args.num_images % class_num == 0
    ), "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print(
            "Generation step {}/{} (cfg={}, cfg_dino={})".format(
                i, num_steps, cfg_label, cfg_dino
            )
        )

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id = (
                i * sampled_images.size(0) * world_size
                + local_rank * sampled_images.size(0)
                + b_id
            )
            if img_id >= args.num_images:
                break
            gen_img = np.round(
                np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
            )
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(
                os.path.join(save_folder, "{}.png".format(str(img_id).zfill(5))),
                gen_img,
            )

    torch.distributed.barrier()

    if misc.get_rank() == 0:
        if (
            hasattr(args, "fid_statistics_file")
            and args.fid_statistics_file is not None
        ):
            fid_statistics_file = args.fid_statistics_file
        else:
            if args.img_size == 256:
                fid_statistics_file = "fid_stats/jit_in256_stats.npz"
            elif args.img_size == 512:
                fid_statistics_file = "fid_stats/jit_in512_stats.npz"
            else:
                raise NotImplementedError(
                    f"No default FID statistics for image size {args.img_size}. "
                    f"Please provide --fid_statistics_file"
                )
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        postfix = "_cfg{}_res{}".format(cfg_label, args.img_size)

        if log_writer is not None:
            log_writer.add_scalar("fid{}".format(postfix), fid, epoch)
            log_writer.add_scalar("is{}".format(postfix), inception_score, epoch)

        cfg_tag = "cfg{}".format(cfg_label)
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log(
                {
                    "eval_{}/fid".format(cfg_tag): fid,
                    "eval_{}/inception_score".format(cfg_tag): inception_score,
                    "eval_{}/cfg_label".format(cfg_tag): cfg_label,
                    "eval_{}/num_sampling_steps".format(
                        cfg_tag
                    ): model_without_ddp.steps,
                    "epoch": epoch,
                }
            )

        print(
            "CFG={} FID: {:.4f}, Inception Score: {:.4f}".format(
                cfg_label, fid, inception_score
            )
        )
        # shutil.rmtree(save_folder)
    else:
        fid = None
        inception_score = None

    torch.distributed.barrier()

    # Restore original cfg_scale
    model_without_ddp.cfg_label = original_cfg_scale

    return fid, inception_score


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    """
    Evaluate with both args.cfg and cfg=1.0 (no CFG), logging metrics separately.
    """

    model_without_ddp.eval()

    # Switch to EMA params
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # Evaluate with the user's configured CFG scale
    print("=" * 60)
    print("Evaluating with cfg_scale={}".format(args.cfg))
    print("=" * 60)
    _evaluate_single_cfg(
        model_without_ddp,
        args,
        epoch,
        batch_size,
        log_writer,
        cfg_label=args.cfg,
        cfg_dino=args.cfg_dino,
    )

    # Evaluate without CFG (cfg_scale=1.0)
    print("=" * 60)
    print("Evaluating with cfg_scale=1.0 (no CFG)")
    print("=" * 60)
    _evaluate_single_cfg(
        model_without_ddp,
        args,
        epoch,
        batch_size,
        log_writer,
        cfg_label=1.0,
        cfg_dino=None,
    )

    # Restore original model weights (back from EMA)
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)


def evaluate_cfg_sweep(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    """
    Sweep over a list of CFG values, evaluating FID/IS for each.
    Both cfg_label and cfg_dino are set to the same value for each step.
    Results are saved to a CSV file under args.output_dir.
    """
    import csv

    cfg_values = [float(v.strip()) for v in args.cfg_sweep.split(",")]
    print("=" * 60)
    print("CFG Sweep: evaluating {} values: {}".format(len(cfg_values), cfg_values))
    print("=" * 60)

    model_without_ddp.eval()

    # Switch to EMA params
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    results = []
    for cfg_val in cfg_values:
        print("=" * 60)
        print("CFG Sweep: evaluating cfg={}".format(cfg_val))
        print("=" * 60)
        fid, inception_score = _evaluate_single_cfg(
            model_without_ddp,
            args,
            epoch,
            batch_size,
            log_writer,
            cfg_label=cfg_val,
            cfg_dino=cfg_val,
        )
        results.append((cfg_val, fid, inception_score))
        print(
            "CFG Sweep: cfg={} -> FID={}, IS={}".format(cfg_val, fid, inception_score)
        )

    # Save results to CSV (only on rank 0)
    if misc.get_rank() == 0:
        csv_path = os.path.join(
            args.output_dir, "cfg_sweep_results_epoch{}.csv".format(epoch)
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cfg_scale", "fid", "inception_score"])
            for cfg_val, fid, inception_score in results:
                writer.writerow([cfg_val, fid, inception_score])
        print("CFG Sweep results saved to {}".format(csv_path))

        # Print summary table
        print("\n" + "=" * 60)
        print("CFG Sweep Summary (epoch {})".format(epoch))
        print("{:<12} {:<12} {:<12}".format("CFG", "FID", "IS"))
        print("-" * 36)
        for cfg_val, fid, inception_score in results:
            print(
                "{:<12.2f} {:<12.4f} {:<12.4f}".format(
                    cfg_val,
                    fid if fid is not None else float("nan"),
                    inception_score if inception_score is not None else float("nan"),
                )
            )
        print("=" * 60)

    # Restore original model weights (back from EMA)
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
