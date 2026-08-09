#!/usr/bin/env python3
"""
Evaluate GenLOB v2 model: generate samples and compute metrics.

Metrics:
1. Physics violation rates (crossed book, ladder, negative volume)
2. Distributional similarity (MSE, MAE, correlation)
3. Spread statistics
4. Sample visualization
"""

import sys
import argparse
import logging
import random
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.models.genlob_v2 import GenLOBv2, GenLOBv2Config
from genlob.models.sde import VPSDE
from genlob.models.sampling import ancestral_sample, ddim_sample
from genlob.utils.physics import (
    compute_projection_distance,
    compute_violation_rate,
    project_to_valid_lob,
)
from genlob.utils.legal_lob import COORDINATE_REPRESENTATIONS, make_lob_transform
from genlob.data.nasdaq import (
    load_nasdaq_multiday,
    NASDAQMultiDayDataset,
    normalized_price_unit,
    parse_snapshot_dataset_id,
    resolve_lob_constraints,
    resolve_price_reference,
    select_evaluation_indices,
)
from genlob.eval.metrics import (
    discrete_lob_validity,
    horizon_fidelity,
    json_safe,
    per_sample_metrics,
    projection_repair_details,
    structural_validity_details,
    stylized_facts,
)
from genlob.utils.device import select_device
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def _load_pyplot():
    """Load the optional plotting stack only when figures are requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, device, weights="ema"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', GenLOBv2Config())
    model = GenLOBv2(config).to(device)

    # Try EMA weights first unless raw model weights are requested.
    if weights == "ema" and 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        logger.info("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Loaded model weights")

    model.eval()
    epoch = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', '?')
    logger.info(f"Checkpoint: epoch={epoch}, val_loss={val_loss}")
    return model, config, ckpt


def resolve_representation(requested: str, ckpt: dict) -> str:
    if requested != "auto":
        return requested
    ckpt_args = ckpt.get("args") or {}
    return ckpt_args.get("representation", "raw")


def resolve_checkpoint_protocol(args, ckpt: dict, dataset: dict) -> None:
    """Resolve preprocessing/constraint protocol without silently changing old checkpoints."""
    ckpt_args = ckpt.get("args") or {}
    if args.action_conditioning == "auto":
        args.action_conditioning = ckpt_args.get("action_conditioning", "proxy")
    if args.price_reference == "auto":
        requested_reference = ckpt_args.get("price_reference", "dataset")
    else:
        requested_reference = args.price_reference
    args.price_reference = resolve_price_reference(
        requested_reference,
        bool(dataset.get("has_raw_states", False)),
    )
    checkpoint_spread = ckpt_args.get("min_spread")
    checkpoint_gap = ckpt_args.get("min_gap")
    args.min_spread, args.min_gap = resolve_lob_constraints(
        dataset.get("normalization"),
        min_spread=args.min_spread if args.min_spread is not None else checkpoint_spread,
        min_gap=args.min_gap if args.min_gap is not None else checkpoint_gap,
    )


def representation_stats_to_device(ckpt: dict, device: torch.device) -> dict | None:
    stats = ckpt.get("representation_stats")
    if not stats:
        return None
    return {
        "mean": stats["mean"].to(device),
        "std": stats["std"].to(device),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or b.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _temporal_corr(real: torch.Tensor, gen: torch.Tensor) -> float:
    vals = []
    for idx in range(min(len(real), 500)):
        r = ((real[idx, 0, 0, :] + real[idx, 0, 10, :]) / 2).numpy()
        g = ((gen[idx, 0, 0, :] + gen[idx, 0, 10, :]) / 2).numpy()
        if r.std() > 1e-8 and g.std() > 1e-8:
            vals.append(np.corrcoef(r, g)[0, 1])
    return float(np.nanmean(vals)) if vals else 0.0


def _summarize_sampler_projection(records: list[dict]) -> dict | None:
    if not records:
        return None

    out = {
        "n_batches": len(records),
        "n_projection_calls": int(sum(r["n_projection_calls"] for r in records)),
        "per_step_projection_l2_mean": float(np.mean([r["per_step_projection_l2_mean"] for r in records])),
        "per_step_projection_l2_max": float(max(r["per_step_projection_l2_max"] for r in records)),
        "cumulative_projection_l2_mean": float(np.mean([r["cumulative_projection_l2_mean"] for r in records])),
        "cumulative_projection_l2_max": float(max(r["cumulative_projection_l2_max"] for r in records)),
    }
    for metric in [
        "crossed_book_rate",
        "ask_ladder_violation",
        "bid_ladder_violation",
        "negative_volume_rate",
    ]:
        mean_key = f"pre_projection_{metric}_mean"
        max_key = f"pre_projection_{metric}_max"
        out[mean_key] = float(np.mean([r[mean_key] for r in records]))
        out[max_key] = float(max(r[max_key] for r in records))
    return out


@torch.no_grad()
def generate_and_evaluate(model, test_loader, sde, device, n_batches=10,
                          use_ddim=True, ddim_steps=50, output_dir="results",
                          projection_mode="exact", representation="raw",
                          min_spread=1e-6, min_gap=0.0,
                          representation_stats=None,
                          sampler_projection_mode="none",
                          price_unit=1.0,
                          normalization=None,
                          save_sequences=True,
                          make_plots=True,
                          evaluation_protocol=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_real = []
    all_gen_raw = []
    all_gen_proj = []
    all_persistence = []
    sampler_projection_records = []

    for i, batch in enumerate(test_loader):
        if n_batches is not None and i >= n_batches:
            break

        if len(batch) == 3:
            history, future, action = batch
            action = action.to(device)
        else:
            history, future = batch
            B = future.shape[0]
            action = torch.zeros(B, model.config.action_dim, device=device)
        if evaluation_protocol and evaluation_protocol.get("action_conditioning") == "none":
            action = torch.zeros_like(action)

        history_lob = history.to(device)
        future_lob = future.to(device)
        transform = (
            make_lob_transform(representation, min_spread=min_spread, gap_eps=min_gap)
            if representation != "raw"
            else None
        )
        if representation in COORDINATE_REPRESENTATIONS:
            assert transform is not None
            history_model = transform.encode(history_lob)
            future_model = transform.encode(future_lob)
            if representation_stats is not None:
                history_model = (history_model - representation_stats["mean"]) / representation_stats["std"]
                future_model = (future_model - representation_stats["mean"]) / representation_stats["std"]
        elif representation == "raw":
            history_model = history_lob
            future_model = future_lob
        else:
            raise ValueError(f"Unknown representation: {representation}")

        # Generate samples
        shape = future_model.shape
        project_fn = None
        step_l2_values = []
        step_pre_violations = []
        if sampler_projection_mode != "none":
            if representation != "raw":
                raise ValueError("sampler_projection_mode is only valid for raw representation")
            if sampler_projection_mode == "exact":
                def project_fn(z):
                    z_cpu = z.detach().cpu()
                    projected_cpu = project_to_valid_lob(
                        z_cpu,
                        min_spread=min_spread,
                        min_gap=min_gap,
                        use_gpu_approx=False,
                    )
                    diff = z_cpu - projected_cpu
                    step_l2_values.append(
                        (diff ** 2).sum(dim=(1, 2, 3)).sqrt()
                    )
                    step_pre_violations.append(
                        compute_violation_rate(z_cpu, min_spread=min_spread, min_gap=min_gap)
                    )
                    return projected_cpu.to(z.device)
            elif sampler_projection_mode == "fast":
                def project_fn(z):
                    projected = project_to_valid_lob(
                        z,
                        min_spread=min_spread,
                        min_gap=min_gap,
                        use_gpu_approx=True,
                    )
                    z_cpu = z.detach().cpu()
                    projected_cpu = projected.detach().cpu()
                    diff = z_cpu - projected_cpu
                    step_l2_values.append(
                        (diff ** 2).sum(dim=(1, 2, 3)).sqrt()
                    )
                    step_pre_violations.append(
                        compute_violation_rate(z_cpu, min_spread=min_spread, min_gap=min_gap)
                    )
                    return projected
            else:
                raise ValueError(f"Unknown sampler_projection_mode: {sampler_projection_mode}")

        if use_ddim:
            generated_model = ddim_sample(
                model, sde, shape, history_model, action,
                ddim_steps=ddim_steps, device=device, project_fn=project_fn,
            )
        else:
            generated_model = ancestral_sample(
                model, sde, shape, history_model, action, device=device, project_fn=project_fn,
            )

        if representation in COORDINATE_REPRESENTATIONS:
            assert transform is not None
            if representation_stats is not None:
                generated_model = generated_model * representation_stats["std"] + representation_stats["mean"]
            generated = transform.decode(generated_model)
        else:
            generated = generated_model

        if step_l2_values:
            l2_stack = torch.stack(step_l2_values)  # [n_projection_calls, B]
            cumulative = l2_stack.sum(dim=0)
            record = {
                "n_projection_calls": int(l2_stack.shape[0]),
                "per_step_projection_l2_mean": float(l2_stack.mean()),
                "per_step_projection_l2_max": float(l2_stack.max()),
                "cumulative_projection_l2_mean": float(cumulative.mean()),
                "cumulative_projection_l2_max": float(cumulative.max()),
            }
            for metric in [
                "crossed_book_rate",
                "ask_ladder_violation",
                "bid_ladder_violation",
                "negative_volume_rate",
            ]:
                values = np.asarray([v[metric] for v in step_pre_violations], dtype=np.float64)
                record[f"pre_projection_{metric}_mean"] = float(values.mean())
                record[f"pre_projection_{metric}_max"] = float(values.max())
            sampler_projection_records.append(record)

        # Apply physics projection. Exact PAVA is preferred for evaluation so
        # the metric reflects the closest valid book instead of the faster
        # in-loop GPU approximation used during DMPP sampling.
        generated_cpu = generated.cpu()
        if projection_mode == "none":
            generated_proj = generated_cpu
        elif projection_mode == "exact":
            generated_proj = project_to_valid_lob(
                generated_cpu,
                min_spread=min_spread,
                min_gap=min_gap,
                use_gpu_approx=False,
            )
        elif projection_mode == "fast":
            generated_proj = project_to_valid_lob(
                generated,
                min_spread=min_spread,
                min_gap=min_gap,
                use_gpu_approx=True,
            ).cpu()
        elif projection_mode == "auto":
            generated_proj = project_to_valid_lob(
                generated,
                min_spread=min_spread,
                min_gap=min_gap,
            ).cpu()
        else:
            raise ValueError(f"Unknown projection_mode: {projection_mode}")

        all_real.append(future_lob.cpu())
        all_gen_raw.append(generated_cpu)
        all_gen_proj.append(generated_proj)
        all_persistence.append(
            history_lob[..., -1:].expand(-1, -1, -1, future_lob.shape[-1]).cpu().clone()
        )

        logger.info(f"Batch {i+1}/{len(test_loader)} generated")

    real = torch.cat(all_real)
    gen_raw = torch.cat(all_gen_raw)
    gen_proj = torch.cat(all_gen_proj)
    persistence = torch.cat(all_persistence)

    logger.info(f"\nTotal samples: {len(real)}")

    # === Metrics ===
    protocol_for_json = dict(evaluation_protocol or {})
    for private_key in [
        "selected_dataset_indices", "selected_start_indices",
        "selected_day_id", "selected_segment_id",
    ]:
        protocol_for_json.pop(private_key, None)
    results = {
        "representation": representation,
        "projection_mode": projection_mode,
        "sampler_projection_mode": sampler_projection_mode,
        "min_spread": min_spread,
        "min_gap": min_gap,
        "evaluation_protocol": protocol_for_json,
        "has_representation_stats": representation_stats is not None,
    }

    # 1. Reconstruction quality
    mse_raw = ((gen_raw - real) ** 2).mean().item()
    mse_proj = ((gen_proj - real) ** 2).mean().item()
    mae_raw = (gen_raw - real).abs().mean().item()
    mae_proj = (gen_proj - real).abs().mean().item()

    results['mse_raw'] = mse_raw
    results['mse_proj'] = mse_proj
    results['mse_projected'] = mse_proj
    results['mae_raw'] = mae_raw
    results['mae_proj'] = mae_proj
    results['mae_projected'] = mae_proj
    logger.info(f"MSE:  raw={mse_raw:.6f}  proj={mse_proj:.6f}")
    logger.info(f"MAE:  raw={mae_raw:.6f}  proj={mae_proj:.6f}")

    # 1b. Projection distortion. Zero violations are only meaningful if the
    # projection is not doing all of the modeling work.
    # Projection distance is always an exact diagnostic distance to the valid
    # manifold. When projection_mode="none", the evaluated sample remains raw,
    # but this still reports whether projection would have changed it.
    if projection_mode == "exact":
        gen_proj_check = gen_proj
    else:
        gen_proj_check = project_to_valid_lob(
            gen_raw.cpu(), min_spread=min_spread, min_gap=min_gap, use_gpu_approx=False,
        )
    proj_dist = compute_projection_distance(gen_raw, gen_proj_check)
    spread_raw = gen_raw[:, 0, 0, :] - gen_raw[:, 0, 10, :]
    spread_proj = gen_proj[:, 0, 0, :] - gen_proj[:, 0, 10, :]
    spread_real = real[:, 0, 0, :] - real[:, 0, 10, :]
    results["projection_distance"] = proj_dist
    results["spread_mean_raw"] = float(spread_raw.mean())
    results["spread_mean_proj"] = float(spread_proj.mean())
    results["spread_mean_projected"] = float(spread_proj.mean())
    results["spread_mean_real"] = float(spread_real.mean())
    results["spread_shift_raw"] = float(spread_raw.mean() - spread_real.mean())
    results["spread_shift_projected"] = float(spread_proj.mean() - spread_real.mean())
    results["spread_mae_proj_vs_raw"] = float((spread_proj - spread_raw).abs().mean())
    results["spread_mae_proj_vs_real"] = float((spread_proj - spread_real).abs().mean())
    sampler_projection_stats = _summarize_sampler_projection(sampler_projection_records)
    if sampler_projection_stats is not None:
        results["sampler_projection_stats"] = sampler_projection_stats
        logger.info(
            "Sampler projection path: calls=%d per_step_l2=%.6f cumulative_l2=%.6f",
            sampler_projection_stats["n_projection_calls"],
            sampler_projection_stats["per_step_projection_l2_mean"],
            sampler_projection_stats["cumulative_projection_l2_mean"],
        )
    logger.info(
        "Projection distance: total=%.6f price=%.6f volume=%.6f",
        proj_dist["total_l2_distance"],
        proj_dist["price_l2_distance"],
        proj_dist["volume_l2_distance"],
    )
    logger.info(
        "Spread mean: real=%.6f raw=%.6f proj=%.6f",
        results["spread_mean_real"],
        results["spread_mean_raw"],
        results["spread_mean_proj"],
    )

    # 2. Physics violations
    viol_raw = compute_violation_rate(gen_raw, min_spread=min_spread, min_gap=min_gap)
    viol_proj = compute_violation_rate(gen_proj, min_spread=min_spread, min_gap=min_gap)
    viol_real = compute_violation_rate(real, min_spread=min_spread, min_gap=min_gap)

    results['violations_raw'] = viol_raw
    results['violations_proj'] = viol_proj
    results['violations_real'] = viol_real
    results["structural_validity_real"] = structural_validity_details(
        real, min_spread, min_gap
    )
    results["structural_validity_raw"] = structural_validity_details(
        gen_raw, min_spread, min_gap
    )
    results["structural_validity_projected"] = structural_validity_details(
        gen_proj, min_spread, min_gap
    )
    results["diagnostic_projection_repair"] = projection_repair_details(
        gen_raw, gen_proj_check, normalization
    )
    results["output_projection_repair"] = projection_repair_details(
        gen_raw, gen_proj, normalization
    )

    logger.info(f"\nPhysics violations:")
    logger.info(f"  {'':20s} {'Real':>10s} {'Generated':>10s} {'Projected':>10s}")
    violation_keys = [
        "crossed_book_rate",
        "ask_ladder_violation",
        "bid_ladder_violation",
        "negative_volume_rate",
        "avg_spread",
    ]
    for key in violation_keys:
        logger.info(f"  {key:20s} {viol_real[key]:10.4f} {viol_raw[key]:10.4f} {viol_proj[key]:10.4f}")

    # 3. Per-channel correlation
    for ch, name in [(0, 'Price'), (1, 'Volume')]:
        real_flat = real[:, ch].reshape(-1).numpy()
        raw_flat = gen_raw[:, ch].reshape(-1).numpy()
        proj_flat = gen_proj[:, ch].reshape(-1).numpy()
        corr_raw = _safe_corr(real_flat, raw_flat)
        corr_proj = _safe_corr(real_flat, proj_flat)
        key = name.lower()
        results[f'corr_{key}_raw'] = corr_raw
        results[f'corr_{key}_projected'] = corr_proj
        # Backward-compatible field: this is the correlation of the evaluated
        # output. For projection_mode="none" it equals raw; otherwise projected.
        results[f'corr_{key}'] = corr_proj
        logger.info(f"Correlation ({name}): raw={corr_raw:.4f} projected={corr_proj:.4f}")

    # 4. Temporal correlation (within each sample)
    temp_corr_raw = _temporal_corr(real, gen_raw)
    temp_corr_proj = _temporal_corr(real, gen_proj)
    results['temporal_correlation_raw'] = temp_corr_raw
    results['temporal_correlation_projected'] = temp_corr_proj
    results['temporal_correlation'] = temp_corr_proj
    logger.info(
        "Temporal correlation (mid path): raw=%.4f projected=%.4f",
        temp_corr_raw,
        temp_corr_proj,
    )

    results["stylized_facts_raw"] = stylized_facts(real, gen_raw, price_unit=price_unit)
    results["stylized_facts_projected"] = stylized_facts(real, gen_proj, price_unit=price_unit)
    results["horizon_fidelity_raw"] = horizon_fidelity(real, gen_raw, persistence)
    results["horizon_fidelity_projected"] = horizon_fidelity(real, gen_proj, persistence)
    results["horizon_fidelity_persistence"] = horizon_fidelity(real, persistence)
    results["discrete_validity_real"] = discrete_lob_validity(real, normalization)
    results["discrete_validity_raw"] = discrete_lob_validity(gen_raw, normalization)
    results["discrete_validity_projected"] = discrete_lob_validity(gen_proj, normalization)
    sample_metrics = per_sample_metrics(
        real,
        gen_raw,
        gen_proj,
        min_spread=min_spread,
        min_gap=min_gap,
        diagnostic_projection=gen_proj_check,
    )
    selected_indices = (evaluation_protocol or {}).get("selected_dataset_indices")
    if selected_indices is not None:
        sample_metrics["dataset_index"] = np.asarray(selected_indices[:len(real)], dtype=np.int64)
    for key in ["selected_start_indices", "selected_day_id", "selected_segment_id"]:
        values = (evaluation_protocol or {}).get(key)
        if values is not None:
            sample_metrics[key.removeprefix("selected_")] = np.asarray(values[:len(real)])
    np.savez_compressed(output_dir / "per_sample_metrics.npz", **sample_metrics)
    if save_sequences:
        np.save(output_dir / "real_future.npy", real.numpy().astype(np.float32, copy=False))
        np.save(output_dir / "generated_raw.npy", gen_raw.numpy().astype(np.float32, copy=False))
        np.save(
            output_dir / "generated_projected.npy",
            gen_proj.numpy().astype(np.float32, copy=False),
        )
        np.save(
            output_dir / "persistence.npy",
            persistence.numpy().astype(np.float32, copy=False),
        )

    # === Visualization ===
    if make_plots:
        plot_samples(real, gen_raw, gen_proj, output_dir)
        plot_distributions(real, gen_proj, output_dir)

    # Save results
    import json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(json_safe(results), f, indent=2, allow_nan=False)
    logger.info(f"\nResults saved to {output_dir}/")

    return results


def get_data_loader(args, raw):
    """Create a data loader matching train_v2 dataset naming."""
    ds = NASDAQMultiDayDataset(
        raw[args.split],
        raw[f"{args.split}_actions"],
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        day_id=raw.get(f"{args.split}_day_id"),
        segment_id=raw.get(f"{args.split}_segment_id"),
        raw_data=raw.get(f"{args.split}_raw"),
        normalization=raw.get("normalization"),
        price_reference=args.price_reference,
    )
    selected = select_evaluation_indices(
        len(ds),
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        sampling_mode=args.sampling_mode,
    )
    subset = Subset(ds, selected.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    protocol = {
        "sampling_mode": args.sampling_mode,
        "dataset_windows": int(len(ds)),
        "selected_windows": int(len(selected)),
        "selected_dataset_indices": selected.tolist(),
        "selection_is_chronological": True,
    }
    selected_starts = ds.starts[selected]
    protocol["selected_start_indices"] = selected_starts.tolist()
    if ds.day_id is not None:
        protocol["selected_day_id"] = ds.day_id[selected_starts].tolist()
    if ds.segment_id is not None:
        protocol["selected_segment_id"] = ds.segment_id[selected_starts].tolist()
    return loader, protocol


def plot_samples(real, gen_raw, gen_proj, output_dir, n_samples=4):
    plt = _load_pyplot()
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3 * n_samples))
    fig.suptitle("LOB Sample Comparison", fontsize=14)

    for i in range(n_samples):
        idx = i * (len(real) // n_samples)

        for j, (data, title) in enumerate([
            (real[idx], "Real"),
            (gen_raw[idx], "Generated (raw)"),
            (gen_proj[idx], "Generated (projected)"),
        ]):
            ax = axes[i, j]
            # Plot ask prices (top) and bid prices (bottom)
            prices = data[0].numpy()  # (20, T)
            T = prices.shape[1]
            t = np.arange(T)

            for k in range(10):
                ax.plot(t, prices[k], 'r-', alpha=0.3 + 0.07 * (10 - k), linewidth=0.8)
                ax.plot(t, prices[10 + k], 'b-', alpha=0.3 + 0.07 * (10 - k), linewidth=0.8)

            if i == 0:
                ax.set_title(title)
            ax.set_ylabel(f"Sample {i+1}")
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved sample_comparison.png")


def plot_distributions(real, gen_proj, output_dir):
    plt = _load_pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Spread distribution
    real_spread = (real[:, 0, 0, :] - real[:, 0, 10, :]).numpy().flatten()
    gen_spread = (gen_proj[:, 0, 0, :] - gen_proj[:, 0, 10, :]).numpy().flatten()
    axes[0, 0].hist(real_spread, bins=50, alpha=0.5, label='Real', density=True)
    axes[0, 0].hist(gen_spread, bins=50, alpha=0.5, label='Generated', density=True)
    axes[0, 0].set_title('Spread Distribution')
    axes[0, 0].legend()

    # Ask L1 price distribution
    axes[0, 1].hist(real[:, 0, 0, :].numpy().flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[0, 1].hist(gen_proj[:, 0, 0, :].numpy().flatten(), bins=50, alpha=0.5, label='Gen', density=True)
    axes[0, 1].set_title('Ask L1 Price')
    axes[0, 1].legend()

    # Bid L1 price
    axes[0, 2].hist(real[:, 0, 10, :].numpy().flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[0, 2].hist(gen_proj[:, 0, 10, :].numpy().flatten(), bins=50, alpha=0.5, label='Gen', density=True)
    axes[0, 2].set_title('Bid L1 Price')
    axes[0, 2].legend()

    # Volume L1
    axes[1, 0].hist(real[:, 1, 0, :].numpy().flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[1, 0].hist(gen_proj[:, 1, 0, :].numpy().flatten(), bins=50, alpha=0.5, label='Gen', density=True)
    axes[1, 0].set_title('Ask L1 Volume')
    axes[1, 0].legend()

    # Mid-price return autocorrelation
    real_mid = ((real[:, 0, 0, :] + real[:, 0, 10, :]) / 2).numpy()
    gen_mid = ((gen_proj[:, 0, 0, :] + gen_proj[:, 0, 10, :]) / 2).numpy()
    real_returns = np.diff(real_mid, axis=1).flatten()
    gen_returns = np.diff(gen_mid, axis=1).flatten()

    axes[1, 1].hist(real_returns, bins=50, alpha=0.5, label='Real', density=True)
    axes[1, 1].hist(gen_returns, bins=50, alpha=0.5, label='Gen', density=True)
    axes[1, 1].set_title('Mid-Price Return Distribution')
    axes[1, 1].legend()

    # Autocorrelation of returns
    def autocorr(x, max_lag=20):
        result = []
        x = x - x.mean()
        if x.size < 2 or x.std() < 1e-12:
            return [float("nan")] * max_lag
        for lag in range(max_lag):
            if lag == 0:
                result.append(1.0)
            else:
                c = _safe_corr(x[:-lag], x[lag:])
                result.append(c)
        return result

    real_acf = autocorr(real_returns[:100000])
    gen_acf = autocorr(gen_returns[:100000])
    axes[1, 2].bar(np.arange(20) - 0.15, real_acf, width=0.3, alpha=0.7, label='Real')
    axes[1, 2].bar(np.arange(20) + 0.15, gen_acf, width=0.3, alpha=0.7, label='Gen')
    axes[1, 2].set_title('Return Autocorrelation')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "distributions.png", dpi=150)
    plt.close()
    logger.info("Saved distributions.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_legal/intc_9day_full_50ep/best.pt")
    parser.add_argument("--dataset", type=str, default="INTC_9day",
                        help="Snapshot dataset id, e.g. INTC_9day; nasdaq_INTC_9day is accepted as an alias")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str,
                        default="results/validation_legal_full/eval")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--sampling-mode", choices=["uniform", "head", "full"], default="uniform",
                        help="Select windows uniformly over the split by default; full evaluates every window.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--projection-mode", choices=["none", "exact", "fast", "auto"], default="exact")
    parser.add_argument("--sampler-projection-mode", choices=["none", "exact", "fast"], default="none",
                        help="Project predicted x0 inside the sampler. Use for in-loop DMPP/PAVA baselines.")
    parser.add_argument(
        "--representation",
        choices=["auto", "raw", "matched_lob", "legal_lob", "legal_lob_ba"],
        default="auto",
    )
    parser.add_argument("--action-conditioning", choices=["auto", "none", "proxy"], default="auto")
    parser.add_argument("--price-reference", choices=["auto", "dataset", "history_last_mid"], default="auto")
    parser.add_argument("--min-spread", type=float, default=None)
    parser.add_argument("--min-gap", type=float, default=None)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible regression runs; set negative to disable")
    parser.add_argument(
        "--no-save-sequences",
        dest="save_sequences",
        action="store_false",
        help="Do not retain real/generated arrays needed for ensemble scores.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip optional Matplotlib figures for batch evaluation runs.",
    )
    parser.set_defaults(save_sequences=True)
    args = parser.parse_args()
    set_seed(None if args.seed < 0 else args.seed)

    device = select_device(args.device)
    logger.info(f"Device: {device}")

    model, config, ckpt = load_model(args.checkpoint, device, weights=args.weights)
    representation = resolve_representation(args.representation, ckpt)
    representation_stats = representation_stats_to_device(ckpt, device)
    logger.info(f"Representation: {representation}")
    if representation_stats is not None:
        logger.info("Loaded representation stats")

    stock, nday = parse_snapshot_dataset_id(args.dataset)
    raw = load_nasdaq_multiday(args.data_dir or "data", stock, nday)
    resolve_checkpoint_protocol(args, ckpt, raw)
    logger.info(
        "Protocol: price_reference=%s min_spread=%.8g min_gap=%.8g sampling=%s",
        args.price_reference,
        args.min_spread,
        args.min_gap,
        args.sampling_mode,
    )

    sde = VPSDE(N=config.num_steps)

    data_loader, evaluation_protocol = get_data_loader(args, raw)
    evaluation_protocol["action_conditioning"] = args.action_conditioning
    price_unit = normalized_price_unit(raw.get("normalization")) or 1.0

    generate_and_evaluate(model, data_loader, sde, device,
                          n_batches=None, ddim_steps=args.ddim_steps,
                          output_dir=args.output_dir,
                          projection_mode=args.projection_mode,
                          representation=representation,
                          min_spread=args.min_spread,
                          min_gap=args.min_gap,
                          representation_stats=representation_stats,
                          sampler_projection_mode=args.sampler_projection_mode,
                          price_unit=price_unit,
                          normalization=raw.get("normalization"),
                          save_sequences=args.save_sequences,
                          make_plots=not args.skip_plots,
                          evaluation_protocol=evaluation_protocol)


if __name__ == "__main__":
    main()
