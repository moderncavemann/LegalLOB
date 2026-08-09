#!/usr/bin/env python3
"""Train GenLOB v2 on the trimmed NASDAQ legal_lob experiment path."""

import os
import sys
import argparse
import contextlib
import logging
import random
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.models.genlob_v2 import GenLOBv2, GenLOBv2Config
from genlob.models.sde import VPSDE, get_loss_fn
from torch.utils.data import DataLoader

from genlob.data.action_templates import make_counterfactual_action
from genlob.utils.physics import (
    action_direction_penalty,
    compute_violation_rate,
    soft_lob_penalty,
)
from genlob.utils.device import select_device
from genlob.utils.legal_lob import (
    COORDINATE_REPRESENTATIONS,
    LegalLOBTransform,
    make_lob_transform,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GenLOB v2")

    # Data
    parser.add_argument("--dataset", type=str, default="INTC_9day",
                        help="Snapshot dataset id, e.g. INTC_9day; nasdaq_INTC_9day is accepted as an alias")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)

    # Model
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--action-conditioning", choices=["none", "proxy"], default="none",
                        help="Formal snapshot experiments use none; proxy is diagnostic only.")
    parser.add_argument(
        "--representation",
        choices=["raw", "matched_lob", "legal_lob", "legal_lob_ba"],
        default="raw",
        help=(
            "Diffusion training space. matched_lob uses unconstrained midpoint/spread/gap/volume "
            "coordinates; legal_lob applies smooth positivity-preserving links; "
            "legal_lob_ba uses a boundary-aware squared-ReLU link for price gaps."
        ),
    )
    parser.add_argument(
        "--price-reference",
        choices=["auto", "dataset", "history_last_mid"],
        default="auto",
        help="Sequence price reference. auto uses history_last_mid when raw states are available.",
    )
    parser.add_argument("--representation-stat-batches", type=int, default=None,
                        help="Train batches used to standardize matched/legal coordinate space; default uses max-train-batches or all")
    parser.add_argument("--representation-eps", type=float, default=1e-6)

    # SDE
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument("--num-scales", type=int, default=100)
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuous SDE loss instead of DDPM")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--cfg-drop", type=float, default=0.5)
    parser.add_argument("--physics-loss-weight", type=float, default=0.0,
                        help="Weight for differentiable LOB validity loss on predicted x0")
    parser.add_argument("--action-contrastive-weight", type=float, default=0.0,
                        help=(
                            "Weight for diagnostic Buy-vs-Sell template loss. "
                            "Keep at 0 unless event-level aggressor-side labels "
                            "are available and validated."
                        ))
    parser.add_argument("--action-margin", type=float, default=0.0,
                        help="Minimum normalized mid-price gap for Buy over Sell")
    parser.add_argument("--allow-diagnostic-action-loss", action="store_true",
                        help="Explicitly allow non-causal action-template loss for diagnostics only.")
    parser.add_argument("--min-spread", type=float, default=None,
                        help="Minimum normalized spread; default reads normalization.json")
    parser.add_argument("--min-gap", type=float, default=None,
                        help="Minimum normalized adjacent-level gap; default reads normalization.json")

    # Saving
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", default=None,
                        help="Resume an interrupted run from last.pt/epoch checkpoint.")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop after this many epochs without fixed-validation improvement; 0 disables.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Optional smoke-test cap for train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Optional smoke-test cap for validation batches")
    parser.add_argument("--fixed-train-order", action="store_true",
                        help="Disable train-loader shuffling, useful for deterministic overfit gates")

    # Hardware
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible training; set negative to disable")
    parser.add_argument("--validation-seed", type=int, default=10000,
                        help="Fixed RNG seed reused for checkpoint-selection validation each epoch")

    return parser.parse_args()


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(args):
    return select_device(args.device)


def get_data_loaders(args):
    data_dir = args.data_dir or "data"
    from genlob.data.nasdaq import (
        load_nasdaq_multiday,
        NASDAQMultiDayDataset,
        parse_snapshot_dataset_id,
        resolve_lob_constraints,
        resolve_price_reference,
    )
    stock, nday = parse_snapshot_dataset_id(args.dataset)
    raw = load_nasdaq_multiday(data_dir, stock, nday)
    args.price_reference = resolve_price_reference(
        args.price_reference,
        bool(raw.get("has_raw_states", False)),
    )
    args.min_spread, args.min_gap = resolve_lob_constraints(
        raw.get("normalization"),
        min_spread=args.min_spread,
        min_gap=args.min_gap,
    )
    loaders = []
    for split in ["train", "val", "test"]:
        ds = NASDAQMultiDayDataset(
            raw[split], raw[f"{split}_actions"],
            args.hist_len, args.pred_len,
            day_id=raw.get(f"{split}_day_id"),
            segment_id=raw.get(f"{split}_segment_id"),
            raw_data=raw.get(f"{split}_raw"),
            normalization=raw.get("normalization"),
            price_reference=args.price_reference,
        )
        loaders.append(DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train" and not args.fixed_train_order),
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        ))
    return tuple(loaders)


def update_ema(ema_model, model, rate=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(rate).add_(param.data, alpha=1 - rate)


def build_lob_transform(args):
    if args.representation == "raw":
        return LegalLOBTransform(min_spread=args.min_spread, gap_eps=args.min_gap)
    return make_lob_transform(
        args.representation,
        min_spread=args.min_spread,
        gap_eps=args.min_gap,
    )


@contextlib.contextmanager
def fixed_validation_rng(seed: int):
    """Make validation noise/timesteps identical across training epochs."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if torch.backends.mps.is_available() and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def checkpoint_payload(
    epoch,
    model,
    ema_model,
    optimizer,
    scheduler,
    val_loss,
    best_val_loss,
    config,
    args,
    rep_stats,
    epochs_without_improvement,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "config": config,
        "args": vars(args),
        "representation_stats": rep_stats,
        "epochs_without_improvement": epochs_without_improvement,
        "rng_state": capture_rng_state(),
    }


def _stats_to_device(stats: dict | None, device: torch.device) -> dict | None:
    if stats is None:
        return None
    return {
        "mean": stats["mean"].to(device),
        "std": stats["std"].to(device),
    }


@torch.no_grad()
def compute_representation_stats(train_loader, args, transform) -> dict | None:
    if args.representation == "raw":
        return None

    max_batches = args.representation_stat_batches
    if max_batches is None:
        max_batches = args.max_train_batches

    stats_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    total = None
    total_sq = None
    count = 0
    for batch_idx, batch in enumerate(stats_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        history, future = batch[0].float(), batch[1].float()
        z = torch.cat([transform.encode(history), transform.encode(future)], dim=-1)
        batch_sum = z.sum(dim=(0, 3), keepdim=True)
        batch_sq = (z ** 2).sum(dim=(0, 3), keepdim=True)
        batch_count = z.shape[0] * z.shape[-1]
        total = batch_sum if total is None else total + batch_sum
        total_sq = batch_sq if total_sq is None else total_sq + batch_sq
        count += batch_count

    if count == 0:
        raise ValueError(f"No batches available to compute {args.representation} representation stats")

    mean = total / count
    var = torch.clamp(total_sq / count - mean ** 2, min=args.representation_eps ** 2)
    std = torch.sqrt(var)
    return {"mean": mean.cpu(), "std": std.cpu()}


def encode_lob_for_model(x: torch.Tensor, args, transform,
                         rep_stats: dict | None = None) -> torch.Tensor:
    if args.representation in COORDINATE_REPRESENTATIONS:
        z = transform.encode(x)
        if rep_stats is not None:
            z = (z - rep_stats["mean"]) / rep_stats["std"]
        return z
    return x


def decode_model_to_lob(z: torch.Tensor, args, transform,
                        rep_stats: dict | None = None) -> torch.Tensor:
    if args.representation in COORDINATE_REPRESENTATIONS:
        if rep_stats is not None:
            z = z * rep_stats["std"] + rep_stats["mean"]
        return transform.decode(z)
    return z


def compute_genlob_loss(model, sde, base_loss_fn, future, history, action, args,
                        transform: LegalLOBTransform, rep_stats: dict | None):
    """Compute diffusion loss plus optional validity/action regularizers."""
    needs_extra = args.physics_loss_weight > 0 or args.action_contrastive_weight > 0
    if not needs_extra:
        return base_loss_fn(model, future, history, action)
    if args.continuous:
        raise ValueError("Extra GenLOB regularizers currently require discrete DDPM training")

    labels = torch.randint(0, sde.N, (future.shape[0],), device=future.device)
    sqrt_alphas = sde.sqrt_alphas_cumprod.to(future.device)
    sqrt_1m = sde.sqrt_1m_alphas_cumprod.to(future.device)
    noise = torch.randn_like(future)
    perturbed = (
        sqrt_alphas[labels, None, None, None] * future
        + sqrt_1m[labels, None, None, None] * noise
    )

    noise_pred = model(perturbed, labels, history, action)
    diffusion_loss = torch.square(noise_pred - noise).reshape(future.shape[0], -1).mean(dim=-1).mean()
    total = diffusion_loss

    if args.physics_loss_weight > 0:
        x0_pred = (perturbed - sqrt_1m[labels, None, None, None] * noise_pred) / (
            sqrt_alphas[labels, None, None, None] + 1e-8
        )
        x0_pred_lob = decode_model_to_lob(x0_pred, args, transform, rep_stats)
        total = total + args.physics_loss_weight * soft_lob_penalty(
            x0_pred_lob,
            min_spread=args.min_spread,
            min_gap=args.min_gap,
        )

    if args.action_contrastive_weight > 0:
        logger.warning(
            "action_contrastive_weight > 0 uses diagnostic action templates. "
            "Do not interpret this as causal Buy/Sell impact unless the dataset "
            "has validated message-level aggressor-side labels."
        )
        buy = make_counterfactual_action(1.0, future.shape[0], future.device, args.dataset, args.action_dim)
        sell = make_counterfactual_action(-1.0, future.shape[0], future.device, args.dataset, args.action_dim)

        # CFG action dropout is useful for guidance, but it weakens this
        # diagnostic template regularizer. Disable it only for these two passes.
        old_drop = model.config.cfg_drop_prob
        model.config.cfg_drop_prob = 0.0
        try:
            noise_buy = model(perturbed, labels, history, buy)
            noise_sell = model(perturbed, labels, history, sell)
        finally:
            model.config.cfg_drop_prob = old_drop

        denom = sqrt_alphas[labels, None, None, None] + 1e-8
        x0_buy = (perturbed - sqrt_1m[labels, None, None, None] * noise_buy) / denom
        x0_sell = (perturbed - sqrt_1m[labels, None, None, None] * noise_sell) / denom
        x0_buy_lob = decode_model_to_lob(x0_buy, args, transform, rep_stats)
        x0_sell_lob = decode_model_to_lob(x0_sell, args, transform, rep_stats)
        total = total + args.action_contrastive_weight * action_direction_penalty(
            x0_buy_lob, x0_sell_lob, margin=args.action_margin,
        )

    return total


def train_epoch(model, ema_model, train_loader, sde, loss_fn, optimizer, scheduler,
                device, args, epoch, transform, rep_stats):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        # Binance returns (history, future, action); others return (history, future)
        if len(batch) == 3:
            history, future, action = batch
            action = action.to(device)
        else:
            history, future = batch
            B = future.shape[0]
            action = torch.zeros(B, args.action_dim, device=device)
        if args.action_conditioning == "none":
            action = torch.zeros_like(action)

        history = history.to(device)
        future = future.to(device)
        batch_stats = _stats_to_device(rep_stats, device)
        history_model = encode_lob_for_model(history, args, transform, batch_stats)
        future_model = encode_lob_for_model(future, args, transform, batch_stats)

        optimizer.zero_grad()
        loss = compute_genlob_loss(
            model, sde, loss_fn, future_model, history_model, action, args, transform,
            batch_stats,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        loss.backward()

        if args.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        update_ema(ema_model, model, args.ema_rate)

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % args.log_every == 0:
            avg = total_loss / n_batches
            logger.info(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] loss={avg:.6f}")

    scheduler.step()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, sde, loss_fn, device, args, transform, rep_stats):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if args.max_val_batches is not None and n_batches >= args.max_val_batches:
            break
        if len(batch) == 3:
            history, future, action = batch
            action = action.to(device)
        else:
            history, future = batch
            B = future.shape[0]
            action = torch.zeros(B, args.action_dim, device=device)
        if args.action_conditioning == "none":
            action = torch.zeros_like(action)

        history = history.to(device)
        future = future.to(device)
        batch_stats = _stats_to_device(rep_stats, device)
        history_model = encode_lob_for_model(history, args, transform, batch_stats)
        future_model = encode_lob_for_model(future, args, transform, batch_stats)

        loss = compute_genlob_loss(
            model, sde, loss_fn, future_model, history_model, action, args, transform,
            batch_stats,
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    if args.action_contrastive_weight > 0 and not args.allow_diagnostic_action_loss:
        raise ValueError(
            "Action contrastive loss is disabled for formal snapshot experiments because labels are "
            "diagnostic proxies. Pass --allow-diagnostic-action-loss only for explicitly non-causal diagnostics."
        )
    set_seed(None if args.seed < 0 else args.seed)
    device = get_device(args)
    logger.info(f"Device: {device}")

    # Data
    logger.info(f"Loading {args.dataset} data...")
    train_loader, val_loader, test_loader = get_data_loaders(args)
    logger.info(f"Train: {len(train_loader.dataset)} samples, "
                f"Val: {len(val_loader.dataset)} samples")
    logger.info(
        "Protocol: price_reference=%s min_spread=%.8g min_gap=%.8g action=%s validation_seed=%d",
        args.price_reference,
        args.min_spread,
        args.min_gap,
        args.action_conditioning,
        args.validation_seed,
    )
    transform = build_lob_transform(args)
    logger.info(f"Representation: {args.representation}")
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous_args = resume_ckpt.get("args") or {}
        for key in [
            "dataset",
            "representation",
            "action_conditioning",
            "hist_len",
            "pred_len",
            "price_reference",
            "epochs",
            "batch_size",
            "num_scales",
        ]:
            if previous_args.get(key) != getattr(args, key):
                raise ValueError(
                    f"Resume protocol mismatch for {key}: checkpoint={previous_args.get(key)} "
                    f"current={getattr(args, key)}"
                )
    rep_stats = (
        resume_ckpt.get("representation_stats")
        if resume_ckpt is not None
        else compute_representation_stats(train_loader, args, transform)
    )
    if rep_stats is not None:
        logger.info(
            "Representation stats: mean=[%.4f, %.4f] std=[%.4f, %.4f]",
            float(rep_stats["mean"].min()),
            float(rep_stats["mean"].max()),
            float(rep_stats["std"].min()),
            float(rep_stats["std"].max()),
        )

    # Model
    config = GenLOBv2Config(
        base_channels=args.base_channels,
        num_layers=args.num_layers,
        action_dim=args.action_dim,
        num_steps=args.num_scales,
        cfg_drop_prob=args.cfg_drop,
    )
    model = GenLOBv2(config).to(device)
    ema_model = GenLOBv2(config).to(device)
    ema_model.load_state_dict(model.state_dict())
    logger.info(f"Model: {model.get_num_params():,} parameters")

    # SDE + Loss
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, N=args.num_scales)
    loss_fn = get_loss_fn(sde, continuous=args.continuous, reduce_mean=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Checkpointing
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history_path = save_dir / "history.json"
    history = []
    if history_path.exists() and args.resume:
        history = json.loads(history_path.read_text()).get("epochs", [])
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        ema_model.load_state_dict(resume_ckpt.get("ema_state_dict", resume_ckpt["model_state_dict"]))
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_val_loss = float(resume_ckpt.get("best_val_loss", resume_ckpt.get("val_loss", float("inf"))))
        epochs_without_improvement = int(resume_ckpt.get("epochs_without_improvement", 0))
        restore_rng_state(resume_ckpt.get("rng_state"))
        logger.info(
            "Resuming from epoch %d with best fixed-validation loss %.6f",
            start_epoch - 1,
            best_val_loss,
        )
    if start_epoch > args.epochs:
        raise ValueError(f"Resume checkpoint already reached epoch {start_epoch - 1} >= requested {args.epochs}")

    logger.info(f"Starting training through epoch {args.epochs}...")

    last_epoch = start_epoch - 1
    val_loss = float("nan")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, ema_model, train_loader, sde, loss_fn, optimizer, scheduler,
            device, args, epoch, transform, rep_stats,
        )

        with fixed_validation_rng(args.validation_seed):
            val_loss = validate(
                model, val_loader, sde, loss_fn, device, args, transform, rep_stats,
            )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train={train_loss:.6f} val={val_loss:.6f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Save checkpoints
        is_best = val_loss < best_val_loss - args.early_stopping_min_delta
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            epoch,
            model,
            ema_model,
            optimizer,
            scheduler,
            val_loss,
            best_val_loss,
            config,
            args,
            rep_stats,
            epochs_without_improvement,
        )
        torch.save(payload, save_dir / "last.pt")
        if is_best:
            torch.save(payload, save_dir / "best.pt")
            logger.info(f"  → New best model saved (val={val_loss:.6f})")

        if epoch % args.save_every == 0:
            torch.save(payload, save_dir / f"epoch_{epoch}.pt")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "learning_rate": lr,
            "elapsed_seconds": elapsed,
            "is_best": is_best,
        })
        history_path.write_text(json.dumps({"epochs": history}, indent=2) + "\n")
        last_epoch = epoch
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            logger.info(
                "Early stopping after %d epochs without fixed-validation improvement",
                epochs_without_improvement,
            )
            break

    # Save final
    final_payload = checkpoint_payload(
        last_epoch,
        model,
        ema_model,
        optimizer,
        scheduler,
        val_loss,
        best_val_loss,
        config,
        args,
        rep_stats,
        epochs_without_improvement,
    )
    torch.save(final_payload, save_dir / "final.pt")
    (save_dir / "completion.json").write_text(json.dumps({
        "status": "complete",
        "last_epoch": last_epoch,
        "requested_epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "early_stopped": last_epoch < args.epochs,
        "fixed_validation_seed": args.validation_seed,
    }, indent=2) + "\n")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
