"""Standard TSF benchmark experiment for HyperbolicTSF on ETTh1.

Replicates the typical long-term forecasting evaluation protocol:
  Dataset:  ETTh1
  seq_len:  96
  pred_len: 96, 192, 336, 720
  Split:    12/4/4 months  ->  split_points=(8640, 11520)
  Features: M  (multivariate predict all)
  Metrics:  MSE, MAE on test set
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsf_data import DataConfig, build_data_bundle
from hypertsf_layers import HyperbolicTSF


# ---------------------------------------------------------------------------
# experiment config
# ---------------------------------------------------------------------------
DATASET = "ETTh1"
DATA_PATH = "datasets/ETT-small/ETTh1.csv"
SEQ_LEN = 96
PRED_LENGTHS = [96, 192, 336, 720]
SPLIT_POINTS = (8640, 11520)          # standard ETTh1 split: 12/4/4 months

# model hyper-params
TANGENT_DIM = 32
HIDDEN_DIM = 64
MANIFOLD = "poincare"
DROPOUT = 0.1
USE_REVIN = True
USE_LINEAR_RESIDUAL = True

# training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5                          # early-stopping patience on val loss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _make_loader(dataset, shuffle: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=_collate,
        pin_memory=(DEVICE == "cuda"),
        drop_last=shuffle,
    )


@torch.no_grad()
def evaluate(model: HyperbolicTSF, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    preds, trues = [], []
    for batch in loader:
        x = batch["x"].to(device)            # [B, seq_len, C_in]
        y = batch["y"].to(device)            # [B, pred_len, C_out]
        pred = model(x)                       # [B, pred_len, C_out]
        preds.append(pred.cpu())
        trues.append(y.cpu())
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    mse = float(((preds - trues) ** 2).mean())
    mae = float((preds - trues).abs().mean())
    return mse, mae


def train_one(
    pred_len: int,
    num_variables: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  pred_len = {pred_len}")
    print(f"{'='*60}")

    model = HyperbolicTSF(
        input_length=SEQ_LEN,
        pred_length=pred_len,
        num_variables=num_variables,
        tangent_dim=TANGENT_DIM,
        hidden_dim=HIDDEN_DIM,
        manifold=MANIFOLD,
        trainable_curvature=True,
        dropout=DROPOUT,
        use_revin=USE_REVIN,
        use_linear_residual=USE_LINEAR_RESIDUAL,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,}  (trainable: {trainable_params:,})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, 
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))

        # --- val ---
        val_mse, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={avg_train:.4f} | val_mse={val_mse:.4f} val_mae={val_mae:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # --- early stopping ---
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s  |  best val MSE = {best_val_loss:.4f}")

    # --- test ---
    model.load_state_dict(best_state)
    model = model.to(device)
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"  >>> TEST  MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    return {
        "pred_len": pred_len,
        "test_mse": round(test_mse, 4),
        "test_mae": round(test_mae, 4),
        "best_val_mse": round(best_val_loss, 4),
        "train_time_s": round(elapsed, 1),
        "total_params": total_params,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET}  ({DATA_PATH})")
    print(f"seq_len={SEQ_LEN}, pred_len={PRED_LENGTHS}")
    print(f"model: manifold={MANIFOLD}, tangent_dim={TANGENT_DIM}, hidden_dim={HIDDEN_DIM}")

    results = []

    for pred_len in PRED_LENGTHS:
        config = DataConfig(
            data_path=DATA_PATH,
            seq_len=SEQ_LEN,
            label_len=0,
            pred_len=pred_len,
            features="M",
            target=None,
            split_points=SPLIT_POINTS,
            scaler="standard",
            add_time_features=False,
            stride=1,
        )
        bundle = build_data_bundle(config)
        num_variables = len(bundle.input_columns)
        print(f"\n  num_variables={num_variables}, input_columns={bundle.input_columns}")
        print(f"  train samples={len(bundle.datasets['train'])}, "
              f"val samples={len(bundle.datasets['val'])}, "
              f"test samples={len(bundle.datasets['test'])}")

        train_loader = _make_loader(bundle.datasets["train"], shuffle=True, batch_size=BATCH_SIZE)
        val_loader = _make_loader(bundle.datasets["val"], shuffle=False, batch_size=BATCH_SIZE)
        test_loader = _make_loader(bundle.datasets["test"], shuffle=False, batch_size=BATCH_SIZE)

        res = train_one(
            pred_len=pred_len,
            num_variables=num_variables,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
        )
        results.append(res)

    # --- summary ---
    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY  -  {DATASET}  (seq_len={SEQ_LEN})")
    print(f"{'='*70}")
    print(f"  {'pred_len':>8}  {'MSE':>8}  {'MAE':>8}  {'val_MSE':>8}  {'time(s)':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['pred_len']:>8}  {r['test_mse']:>8.4f}  {r['test_mae']:>8.4f}  "
              f"{r['best_val_mse']:>8.4f}  {r['train_time_s']:>8.1f}")

    # save results
    out_path = Path(__file__).resolve().parent / "experiment_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET,
                "seq_len": SEQ_LEN,
                "model": "HyperbolicTSF",
                "manifold": MANIFOLD,
                "tangent_dim": TANGENT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
