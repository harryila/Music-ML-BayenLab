"""PyTorch Lightning trainer for MIDI2ScoreTransformer.

Upstream MIDI2ScoreTransformer ships dataset/inference code but no training
loop. This module provides:
    - A training_step / validation_step that adds weighted CE losses across
      all output streams + BCE on the pad stream.
    - configure_optimizers with AdamW + cosine schedule + linear warmup.
    - LR-range test helper (LRFinder-style sweep) for the sanity check before
      committing to a long pretrain run.
    - A `fit_pdmx` and `fit_asap` driver to run Stage A and Stage B.

Loss weights and ignore-index defaults come from `config.FEATURES`. The pad
stream is special-cased as binary (sigmoid -> BCE) since it gates whether each
output position is a real note.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Local imports
from config import FEATURES, MyModelConfig
from models.roformer import Roformer
from utils import pad_batch

# Compat shim (newer transformers): the released ckpt's MyModelConfig predates
# the `_attn_implementation_internal` attribute that current transformers expect
# to read during model construction. Provide a default + allow safe unpickling.
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

log = logging.getLogger(__name__)

device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def collate_fn(batch):
    """Pad a list of (input_stream, output_stream) tuples to a common length."""
    inputs = [b[0] for b in batch]
    outputs = [b[1] for b in batch]
    return pad_batch([{k: v.unsqueeze(0) for k, v in s.items()} for s in inputs]), \
           pad_batch([{k: v.unsqueeze(0) for k, v in s.items()} for s in outputs])


class TrainableRoformer(Roformer):
    """Adds Lightning training methods to the existing Roformer model."""

    def __init__(self, *args, learning_rate: float = 3e-4, weight_decay: float = 0.2,
                 warmup_steps: int = 1000, total_steps: int = 40000,
                 betas: tuple = (0.9, 0.999), input_dropout: float = 0.75,
                 unconditional_dropout: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # Save extra training hparams
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps
        self._betas = betas
        self._input_dropout = input_dropout
        self._unconditional_dropout = unconditional_dropout

    def _maybe_drop_input(self, input_streams: dict) -> dict:
        """Apply token-position dropout (input_dropout) and full-context dropout
        (unconditional_dropout) during training. Inference: no-op."""
        if not self.training:
            return input_streams
        if self._unconditional_dropout > 0 and torch.rand(1).item() < self._unconditional_dropout:
            zeroed = {k: torch.zeros_like(v) for k, v in input_streams.items()}
            zeroed["pad"] = input_streams["pad"]  # keep mask so attention works
            return zeroed
        if self._input_dropout > 0:
            mask_shape = (input_streams["pitch"].shape[0], input_streams["pitch"].shape[1], 1)
            keep = (torch.rand(mask_shape, device=input_streams["pitch"].device) > self._input_dropout).float()
            return {k: (v * keep if v.dim() == 3 else v) for k, v in input_streams.items()}
        return input_streams

    def _maybe_drop_decoder(self, output_streams: dict) -> dict:
        """Drop tokens fed to the decoder. Without this, the decoder learns the
        trivial identity (decoder_in == target), and at inference time when
        the start token is all-zero it predicts all-zeros for everything.
        Match the released checkpoint's input_dropout=0.75 here too."""
        if not self.training or self._input_dropout <= 0:
            return output_streams
        T = output_streams["pitch"].shape[1]
        B = output_streams["pitch"].shape[0]
        device = output_streams["pitch"].device
        keep = (torch.rand((B, T, 1), device=device) > self._input_dropout).float()
        out = {}
        for k, v in output_streams.items():
            out[k] = v * keep if v.dim() == 3 else v
        return out

    def _compute_loss(self, pred: dict, target: dict) -> tuple[torch.Tensor, dict]:
        """Weighted CE per stream + BCE for pad."""
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred[list(pred.keys())[0]].device)

        # Position mask: 1 where the target is a real note
        # target['pad'] shape: (B, T) long -> 1.0 for real, 0.0 for padded
        if target["pad"].dim() == 2:
            real_mask = target["pad"].float()
        else:
            real_mask = target["pad"].float().squeeze(-1)
        # BCE on pad
        if "pad" in pred:
            pad_logits = pred["pad"].squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(pad_logits, real_mask)
            losses["pad"] = bce
            total = total + bce

        for stream, conf in FEATURES.items():
            if stream == "pad" or stream not in pred:
                continue
            if stream not in target:
                continue
            logits = pred[stream]
            tgt_onehot = target[stream]
            if tgt_onehot.dim() < 3:
                continue
            # Convert one-hot to class index. Padded positions also have argmax=0,
            # but ignore_index masks them out via real_mask.
            tgt_idx = tgt_onehot.argmax(-1)  # (B, T)
            # Apply ignore at padded positions
            ignore_idx = conf.get("ignore_index", -100)
            if ignore_idx is None:
                ignore_idx = -100
            tgt_masked = tgt_idx.clone()
            if ignore_idx >= 0:
                tgt_masked[real_mask == 0] = ignore_idx
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_masked.reshape(-1),
                    ignore_index=ignore_idx,
                )
            else:
                # use boolean mask
                B, T, V = logits.shape
                lr = logits.reshape(-1, V)
                tr = tgt_masked.reshape(-1)
                m = real_mask.reshape(-1) > 0
                if m.sum() == 0:
                    ce = torch.tensor(0.0, device=logits.device)
                else:
                    ce = F.cross_entropy(lr[m], tr[m])
            losses[stream] = ce
            # Defensive: when a batch has 100% ignore-class for a given stream
            # (e.g. the entire piece has hand=2 because Part IDs lack staff hints),
            # F.cross_entropy returns NaN. Skip that stream rather than poison total.
            if not torch.isnan(ce):
                total = total + conf["loss_weight"] * ce
        return total, losses

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        input_streams, output_streams = batch
        input_streams = self._maybe_drop_input(input_streams)
        # Drop decoder input separately to prevent the decoder from learning
        # the trivial identity mapping (decoder_in == target).
        decoder_in = self._maybe_drop_decoder(output_streams)
        pred = self.forward(input_streams=input_streams, output_streams=decoder_in)
        loss, parts = self._compute_loss(pred, output_streams)
        self.log("train/total", loss, on_step=True, prog_bar=True)
        for k, v in parts.items():
            self.log(f"train/{k}", v, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        input_streams, output_streams = batch
        pred = self.forward(input_streams=input_streams, output_streams=output_streams)
        loss, parts = self._compute_loss(pred, output_streams)
        self.log("val/total", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self._lr,
            betas=self._betas,
            weight_decay=self._weight_decay,
        )
        # Linear warmup + cosine decay
        warmup = self._warmup_steps
        total = max(self._total_steps, warmup + 1)
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total - warmup))
            progress = min(1.0, progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def _new_model(learning_rate: float = 3e-4, total_steps: int = 40000,
               warmup_steps: int = 1000, **kwargs) -> TrainableRoformer:
    """Create a fresh model matching the released checkpoint's architecture."""
    enc = MyModelConfig(
        is_decoder=False, is_autoregressive=False,
        num_hidden_layers=4, hidden_size=512, num_attention_heads=8,
        intermediate_size=1536, max_position_embeddings=1536,
        embedding_size=512, hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
        rotary_value=False, positional_encoding="RoPE",
    )
    dec = MyModelConfig(
        is_decoder=True, is_autoregressive=False, add_cross_attention=True,
        num_hidden_layers=4, hidden_size=512, num_attention_heads=8,
        intermediate_size=1536, max_position_embeddings=1536,
        embedding_size=512, hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
        rotary_value=False, positional_encoding="RoPE",
    )
    hp = {
        "components": ["encoder", "decoder"],
        "domains": {"in": "midi", "out": "mxl"},
    }
    model = TrainableRoformer(
        enc_configuration=enc, dec_configuration=dec, hyperparameters=hp,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        **kwargs,
    )
    return model


def load_pretrained_init(ckpt_path: Optional[str] = None) -> Optional[TrainableRoformer]:
    """Initialise from a released checkpoint (warm-start) instead of from scratch."""
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return None
    base = Roformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    enc = base.enc_config
    dec = base.dec_config
    hp = base.hyperparameters
    model = TrainableRoformer(enc_configuration=enc, dec_configuration=dec, hyperparameters=hp)
    model.load_state_dict(base.state_dict(), strict=False)
    return model


def lr_range_test(model: TrainableRoformer, loader: DataLoader,
                  start_lr: float = 1e-6, end_lr: float = 1e-3,
                  num_steps: int = 200, out_path: Path | None = None) -> Path:
    """LR-range test (LRFinder-style). Linearly increases LR over num_steps and
    records training loss; you pick the LR at the steepest descent.
    """
    import csv
    model.train()
    device = torch.device(device_str)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=start_lr, betas=(0.9, 0.999), weight_decay=0.2)
    if out_path is None:
        out_path = Path("checkpoints/lr_range_test.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    step = 0
    for batch in loader:
        if step >= num_steps:
            break
        input_streams, output_streams = batch
        input_streams = {k: v.to(device) for k, v in input_streams.items()}
        output_streams = {k: v.to(device) for k, v in output_streams.items()}
        # Linear LR sweep in log space
        prog = step / max(num_steps - 1, 1)
        lr = math.exp(math.log(start_lr) + prog * (math.log(end_lr) - math.log(start_lr)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        pred = model.forward(input_streams=input_streams, output_streams=output_streams)
        loss, _ = model._compute_loss(pred, output_streams)
        loss.backward()
        opt.step()
        rows.append({"step": step, "lr": lr, "loss": float(loss.detach().cpu())})
        log.info("[LR test] step %d lr=%.2e loss=%.4f", step, lr, float(loss.detach().cpu()))
        step += 1

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lr", "loss"])
        w.writeheader()
        w.writerows(rows)
    return out_path


def make_pdmx_loaders(manifest_csv: str, seq_length: int = 512,
                      batch_size: int = 8, num_workers: int = 4):
    from pdmx_dataset import PDMXDataset
    train_ds = PDMXDataset(manifest_csv, split="train", seq_length=seq_length,
                           padding="per-beat",
                           augmentations={"transpose": True, "random_crop": True,
                                          "tempo_jitter": (0.8, 1.2),
                                          "onset_jitter": 0.05,
                                          "random_shift": 8,
                                          "velocity_jitter": 5})
    val_ds = PDMXDataset(manifest_csv, split="validation", seq_length=seq_length,
                         padding="per-beat", augmentations={})
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader


def fit(stage: str, manifest_csv: str, learning_rate: float, max_epochs: int,
        out_dir: str, init_ckpt: Optional[str] = None,
        batch_size: int = 8, seq_length: int = 512, num_workers: int = 4,
        precision: str = "16-mixed") -> str:
    train_loader, val_loader = make_pdmx_loaders(manifest_csv, seq_length, batch_size, num_workers)

    steps_per_epoch = max(1, len(train_loader.dataset) // batch_size)
    total_steps = steps_per_epoch * max_epochs

    if init_ckpt and os.path.exists(init_ckpt):
        model = load_pretrained_init(init_ckpt)
        if model is None:
            model = _new_model(learning_rate=learning_rate,
                               warmup_steps=max(100, total_steps // 50),
                               total_steps=total_steps)
        else:
            model._lr = learning_rate
            model._warmup_steps = max(100, total_steps // 50)
            model._total_steps = total_steps
    else:
        model = _new_model(learning_rate=learning_rate,
                           warmup_steps=max(100, total_steps // 50),
                           total_steps=total_steps)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir_p), filename=f"{stage}-{{epoch:02d}}-{{val/total:.4f}}",
        monitor="val/total", mode="min", save_top_k=2, save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt_cb, lr_cb],
        accelerator="auto",
        devices="auto",
        precision=precision if device_str == "cuda" else "32-true",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        default_root_dir=str(out_dir_p),
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    final_ckpt = ckpt_cb.best_model_path or os.path.join(str(out_dir_p), "last.ckpt")
    log.info("Stage %s best ckpt: %s", stage, final_ckpt)
    return final_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lr_test", help="LR-range test on a few steps")
    p.add_argument("--manifest", required=True)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--start-lr", type=float, default=1e-6)
    p.add_argument("--end-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("checkpoints/lr_range_test.csv"))

    p = sub.add_parser("fit", help="Train a stage")
    p.add_argument("--stage", choices=["pretrain_pdmx", "finetune_asap"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision", default="16-mixed")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.cmd == "lr_test":
        train_loader, _ = make_pdmx_loaders(args.manifest, args.seq_length, args.batch_size, num_workers=0)
        model = _new_model(learning_rate=1e-4, total_steps=args.num_steps,
                           warmup_steps=max(1, args.num_steps // 50))
        out = lr_range_test(model, train_loader, start_lr=args.start_lr, end_lr=args.end_lr,
                            num_steps=args.num_steps, out_path=args.out)
        print(f"Saved LR sweep: {out}")
    elif args.cmd == "fit":
        ckpt = fit(args.stage, args.manifest, args.lr, args.max_epochs, args.out_dir,
                   init_ckpt=args.init_ckpt, batch_size=args.batch_size,
                   seq_length=args.seq_length, num_workers=args.num_workers,
                   precision=args.precision)
        print(f"Best ckpt: {ckpt}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
