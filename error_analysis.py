"""Error-driven analysis: diagnose which test samples the model fails on and why.

Collects per-sample errors and decomposes them by:
  1. Prediction horizon (which future steps are hardest)
  2. Variable (which variables are hardest)
  3. Input characteristics (volatility, trend, regime)
  4. Time of day / day of week (seasonal patterns)
  5. Residual vs direct path contribution per sample
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsf_data import DataConfig, build_data_bundle
from hypertsf_layers import HyperbolicTSF
from error_analysis_utils import decompose_head_forecasts


DATASET = "ETTh1"
DATA_PATH = "datasets/ETT-small/ETTh1.csv"
SEQ_LEN = 96
PRED_LEN = 96          # start with the primary horizon
SPLIT_POINTS = (8640, 11520)

TANGENT_DIM = 32
HIDDEN_DIM = 64
MANIFOLD = "poincare"
DROPOUT = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train_and_save_best(pred_len, num_variables, train_loader, val_loader):
    """Train model, return best checkpoint state dict."""
    model = HyperbolicTSF(
        input_length=SEQ_LEN,
        pred_length=pred_len,
        num_variables=num_variables,
        tangent_dim=TANGENT_DIM,
        hidden_dim=HIDDEN_DIM,
        manifold=MANIFOLD,
        dropout=DROPOUT,
        use_revin=True,
        use_linear_residual=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))

        # val
        model.eval()
        with torch.no_grad():
            val_preds, val_trues = [], []
            for batch in val_loader:
                x = batch["x"].to(DEVICE)
                y = batch["y"].to(DEVICE)
                pred = model(x)
                val_preds.append(pred.cpu())
                val_trues.append(y.cpu())
            val_preds = torch.cat(val_preds, dim=0)
            val_trues = torch.cat(val_trues, dim=0)
            val_mse = float(((val_preds - val_trues) ** 2).mean())

        scheduler.step(val_mse)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"  Epoch {epoch:02d}/{EPOCHS} | train={avg_train:.4f} | val_mse={val_mse:.4f}")

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model


@torch.no_grad()
def collect_errors(model, test_loader, pred_len, num_variables):
    """Run model on test set and collect detailed per-sample diagnostics."""
    model.eval()

    all_preds, all_trues, all_inputs = [], [], []
    all_direct_forecasts, all_residual_forecasts = [], []
    all_direct_contributions, all_residual_contributions = [], []
    all_bases, all_reconstructed_preds = [], []
    all_fusion_gates, all_var_weights = [], []
    all_start_indices = []

    for batch in test_loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        out = model(x, return_aux=True)

        all_preds.append(out["prediction"].cpu())
        all_trues.append(y.cpu())
        all_inputs.append(x.cpu())
        components = decompose_head_forecasts(
            model,
            out["head"]["direct"],
            out["head"]["residual"],
            out["revin_state"],
        )
        all_direct_forecasts.append(components["direct_forecast"].cpu())
        all_residual_forecasts.append(components["residual_forecast"].cpu())
        all_direct_contributions.append(
            components["direct_contribution"].cpu()
        )
        all_residual_contributions.append(
            components["residual_contribution"].cpu()
        )
        all_bases.append(components["base_forecast"].cpu())
        all_reconstructed_preds.append(
            components["reconstructed_prediction"].cpu()
        )
        all_fusion_gates.append(out["encoder"]["fusion_gate"].cpu())
        all_var_weights.append(out["encoder"]["variable_weights"].cpu())
        if "start_idx" in batch:
            all_start_indices.append(batch["start_idx"])

    preds = torch.cat(all_preds, dim=0)
    trues = torch.cat(all_trues, dim=0)
    inputs = torch.cat(all_inputs, dim=0)
    direct_forecast = torch.cat(all_direct_forecasts, dim=0)
    residual_forecast = torch.cat(all_residual_forecasts, dim=0)
    direct_contribution = torch.cat(all_direct_contributions, dim=0)
    residual_contribution = torch.cat(all_residual_contributions, dim=0)
    base_forecast = torch.cat(all_bases, dim=0)
    reconstructed_prediction = torch.cat(all_reconstructed_preds, dim=0)
    fusion_gates = torch.cat(all_fusion_gates, dim=0)
    var_weights = torch.cat(all_var_weights, dim=0)

    if all_start_indices:
        start_indices = torch.cat(all_start_indices, dim=0)
    else:
        start_indices = torch.arange(len(preds))

    return {
        "preds": preds,
        "trues": trues,
        "inputs": inputs,
        "direct_forecast": direct_forecast,
        "residual_forecast": residual_forecast,
        "direct_contribution": direct_contribution,
        "residual_contribution": residual_contribution,
        "base_forecast": base_forecast,
        "reconstructed_prediction": reconstructed_prediction,
        "fusion_gates": fusion_gates,
        "var_weights": var_weights,
        "start_indices": start_indices,
    }


def analyze_errors(data, pred_len, num_variables, variable_names):
    """Comprehensive error analysis across multiple dimensions."""
    preds = data["preds"]
    trues = data["trues"]
    inputs = data["inputs"]
    direct_forecast = data["direct_forecast"]
    residual_forecast = data["residual_forecast"]
    fusion_gates = data["fusion_gates"]
    var_weights = data["var_weights"]

    n_samples = preds.size(0)
    errors = (preds - trues) ** 2          # [N, pred_len, C]
    abs_errors = (preds - trues).abs()     # [N, pred_len, C]

    # ---- 1. Per-sample MSE ----
    sample_mse = errors.mean(dim=(1, 2))   # [N]

    # ---- 2. Per-horizon MSE (which future steps are hardest) ----
    horizon_mse = errors.mean(dim=(0, 2))  # [pred_len]

    # ---- 3. Per-variable MSE (which variables are hardest) ----
    variable_mse = errors.mean(dim=(0, 1))  # [C]

    # ---- 4. Per-variable per-horizon MSE (interaction) ----
    var_horizon_mse = errors.mean(dim=0)    # [pred_len, C]

    # ---- 5. Input characteristics correlation ----
    # Volatility: std of input window
    input_vol = inputs.std(dim=1)            # [N, C]
    input_mean = inputs.mean(dim=1)          # [N, C]

    # Trend: slope of last portion
    half = inputs.size(1) // 2
    first_half = inputs[:, :half, :].mean(dim=1)
    second_half = inputs[:, half:, :].mean(dim=1)
    input_trend = second_half - first_half   # [N, C]

    # Recent volatility (last 24 steps)
    recent_vol = inputs[:, -24:, :].std(dim=1)  # [N, C]

    # Sample-level aggregated
    sample_vol = input_vol.mean(dim=1)       # [N]
    sample_trend = input_trend.mean(dim=1)   # [N]
    sample_recent_vol = recent_vol.mean(dim=1)  # [N]

    # ---- 6. Direct vs residual path analysis ----
    direct_errors = (direct_forecast - trues) ** 2
    residual_errors = (residual_forecast - trues) ** 2

    # Which path is "winning" per sample
    direct_mse_per_sample = direct_errors.mean(dim=(1, 2))
    residual_mse_per_sample = residual_errors.mean(dim=(1, 2))

    # ---- 7. Error correlation with fusion gate ----
    gate_mean_per_sample = fusion_gates.mean(dim=(1, 2))  # [N]

    # ---- 8. Temporal pattern analysis (group by start_idx) ----
    # For hourly data, group by hour of day and day of week
    start_indices = data["start_indices"]

    results = {
        "n_samples": n_samples,
        "overall_mse": float(sample_mse.mean()),
        "overall_mae": float(abs_errors.mean()),

        # per-horizon
        "horizon_mse": horizon_mse.tolist(),

        # per-variable
        "variable_mse": variable_mse.tolist(),
        "variable_names": variable_names,

        # per-variable per-horizon
        "var_horizon_mse": var_horizon_mse.tolist(),

        # correlation analysis
        "sample_mse": sample_mse.tolist(),
        "sample_volatility": sample_vol.tolist(),
        "sample_trend": sample_trend.tolist(),
        "sample_recent_vol": sample_recent_vol.tolist(),
        "sample_direct_mse": direct_mse_per_sample.tolist(),
        "sample_residual_mse": residual_mse_per_sample.tolist(),
        "sample_gate_mean": gate_mean_per_sample.tolist(),

        # start indices for temporal analysis
        "start_indices": start_indices.tolist(),

        # top-K worst and best samples
        "top10_worst_idx": sample_mse.topk(10).indices.tolist(),
        "top10_worst_mse": sample_mse.topk(10).values.tolist(),
        "top10_best_idx": sample_mse.topk(10, largest=False).indices.tolist(),
        "top10_best_mse": sample_mse.topk(10, largest=False).values.tolist(),

        # quantile analysis
        "mse_percentiles": {
            "p10": float(sample_mse.quantile(0.10)),
            "p25": float(sample_mse.quantile(0.25)),
            "p50": float(sample_mse.quantile(0.50)),
            "p75": float(sample_mse.quantile(0.75)),
            "p90": float(sample_mse.quantile(0.90)),
            "p95": float(sample_mse.quantile(0.95)),
            "p99": float(sample_mse.quantile(0.99)),
        },
    }

    return results


def compute_correlations(results):
    """Compute Pearson correlations between error and input characteristics."""
    sample_mse = np.array(results["sample_mse"])
    sample_vol = np.array(results["sample_volatility"])
    sample_trend = np.array(results["sample_trend"])
    sample_recent_vol = np.array(results["sample_recent_vol"])
    sample_gate = np.array(results["sample_gate_mean"])
    sample_direct_mse = np.array(results["sample_direct_mse"])
    sample_residual_mse = np.array(results["sample_residual_mse"])

    def pearson(x, y):
        x = x - x.mean()
        y = y - y.mean()
        num = (x * y).sum()
        den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        return float(num / den) if den > 1e-12 else 0.0

    return {
        "error_vs_volatility": pearson(sample_mse, sample_vol),
        "error_vs_trend": pearson(sample_mse, sample_trend),
        "error_vs_recent_vol": pearson(sample_mse, sample_recent_vol),
        "error_vs_gate": pearson(sample_mse, sample_gate),
        "error_vs_direct_mse": pearson(sample_mse, sample_direct_mse),
        "error_vs_residual_mse": pearson(sample_mse, sample_residual_mse),
        "gate_vs_volatility": pearson(sample_gate, sample_vol),
        "gate_vs_trend": pearson(sample_gate, sample_trend),
        "direct_vs_residual_corr": pearson(sample_direct_mse, sample_residual_mse),
    }


def main():
    print(f"=== Error-Driven Analysis ===")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET}, pred_len={PRED_LEN}")

    # Load data
    config = DataConfig(
        data_path=DATA_PATH,
        seq_len=SEQ_LEN,
        label_len=0,
        pred_len=PRED_LEN,
        features="M",
        target=None,
        split_points=SPLIT_POINTS,
        scaler="standard",
        add_time_features=False,
        stride=1,
    )
    bundle = build_data_bundle(config)
    num_variables = len(bundle.input_columns)
    variable_names = bundle.input_columns

    train_loader = DataLoader(
        bundle.datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        bundle.datasets["val"], batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        bundle.datasets["test"], batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate,
    )

    # Train
    print(f"\n--- Training ---")
    model = train_and_save_best(PRED_LEN, num_variables, train_loader, val_loader)

    # Collect errors
    print(f"\n--- Collecting errors on test set ---")
    data = collect_errors(model, test_loader, PRED_LEN, num_variables)
    print(f"  Test samples: {data['preds'].size(0)}")

    # Analyze
    print(f"\n--- Analyzing errors ---")
    results = analyze_errors(data, PRED_LEN, num_variables, variable_names)
    correlations = compute_correlations(results)

    # ---- Print results ----
    print(f"\n{'='*70}")
    print(f"  ERROR ANALYSIS RESULTS")
    print(f"{'='*70}")

    print(f"\n  Overall: MSE={results['overall_mse']:.4f}, MAE={results['overall_mae']:.4f}")
    print(f"  Samples: {results['n_samples']}")

    # MSE percentiles
    print(f"\n  MSE Distribution (percentiles):")
    for k, v in results["mse_percentiles"].items():
        print(f"    {k:>4}: {v:.4f}")

    # Per-variable
    print(f"\n  Per-Variable MSE (sorted):")
    var_mse_pairs = list(zip(variable_names, results["variable_mse"]))
    var_mse_pairs.sort(key=lambda x: x[1], reverse=True)
    for name, mse in var_mse_pairs:
        print(f"    {name:>8}: {mse:.4f}")

    # Per-horizon (show key points)
    horizon_mse = results["horizon_mse"]
    print(f"\n  Per-Horizon MSE (key steps):")
    key_steps = [0, 1, 5, 11, 23, 47, 71, 95] if PRED_LEN >= 96 else list(range(PRED_LEN))
    for step in key_steps:
        if step < len(horizon_mse):
            print(f"    step {step:>3}: {horizon_mse[step]:.4f}")
    print(f"    first_24_avg: {np.mean(horizon_mse[:24]):.4f}")
    print(f"    last_24_avg:  {np.mean(horizon_mse[-24:]):.4f}")
    print(f"    growth_rate:  {horizon_mse[-1] / horizon_mse[0]:.2f}x")

    # Correlations
    print(f"\n  Correlations:")
    for k, v in correlations.items():
        print(f"    {k:>30}: {v:+.4f}")

    # Top worst/best samples
    print(f"\n  Top 10 Worst Samples (by MSE):")
    for idx, mse in zip(results["top10_worst_idx"], results["top10_worst_mse"]):
        vol = results["sample_volatility"][idx]
        trend = results["sample_trend"][idx]
        gate = results["sample_gate_mean"][idx]
        direct_mse = results["sample_direct_mse"][idx]
        res_mse = results["sample_residual_mse"][idx]
        print(f"    sample {idx:>4}: MSE={mse:.4f} vol={vol:.3f} trend={trend:.3f} "
              f"gate={gate:.3f} direct_mse={direct_mse:.4f} res_mse={res_mse:.4f}")

    print(f"\n  Top 10 Best Samples (by MSE):")
    for idx, mse in zip(results["top10_best_idx"], results["top10_best_mse"]):
        vol = results["sample_volatility"][idx]
        trend = results["sample_trend"][idx]
        gate = results["sample_gate_mean"][idx]
        direct_mse = results["sample_direct_mse"][idx]
        res_mse = results["sample_residual_mse"][idx]
        print(f"    sample {idx:>4}: MSE={mse:.4f} vol={vol:.3f} trend={trend:.3f} "
              f"gate={gate:.3f} direct_mse={direct_mse:.4f} res_mse={res_mse:.4f}")

    # Worst vs best comparison
    worst_mse_avg = np.mean(results["top10_worst_mse"])
    best_mse_avg = np.mean(results["top10_best_mse"])
    worst_indices = results["top10_worst_idx"]
    best_indices = results["top10_best_idx"]

    worst_vol_avg = np.mean([results["sample_volatility"][i] for i in worst_indices])
    best_vol_avg = np.mean([results["sample_volatility"][i] for i in best_indices])
    worst_trend_avg = np.mean([results["sample_trend"][i] for i in worst_indices])
    best_trend_avg = np.mean([results["sample_trend"][i] for i in best_indices])
    worst_gate_avg = np.mean([results["sample_gate_mean"][i] for i in worst_indices])
    best_gate_avg = np.mean([results["sample_gate_mean"][i] for i in best_indices])

    print(f"\n  Worst vs Best Comparison:")
    print(f"    {'':>20} {'Worst-10':>12} {'Best-10':>12} {'Ratio':>8}")
    print(f"    {'MSE':>20} {worst_mse_avg:>12.4f} {best_mse_avg:>12.4f} {worst_mse_avg / best_mse_avg:>8.1f}x")
    print(f"    {'Volatility':>20} {worst_vol_avg:>12.4f} {best_vol_avg:>12.4f} {worst_vol_avg / best_vol_avg:>8.2f}x")
    print(f"    {'Trend':>20} {worst_trend_avg:>12.4f} {best_trend_avg:>12.4f} {abs(worst_trend_avg / best_trend_avg) if abs(best_trend_avg) > 1e-6 else float('inf'):>8.2f}x")
    print(f"    {'Fusion Gate':>20} {worst_gate_avg:>12.4f} {best_gate_avg:>12.4f}")

    # Direct vs residual per variable
    print(f"\n  Direct vs Residual Path per Variable:")
    reconstruction_error = (
        data["reconstructed_prediction"] - data["preds"]
    ).abs().max().item()
    print(f"\n  Branch reconstruction max error: {reconstruction_error:.3e}")

    # Direct and residual are now evaluated as standalone forecasts after
    # valid RevIN denormalization, rather than in incompatible coordinates.
    for var_idx, name in enumerate(variable_names):
        direct_var_mse = ((data["direct_forecast"][:, :, var_idx] - data["trues"][:, :, var_idx]) ** 2).mean().item()
        residual_var_mse = ((data["residual_forecast"][:, :, var_idx] - data["trues"][:, :, var_idx]) ** 2).mean().item()
        pred_var_mse = ((data["preds"][:, :, var_idx] - data["trues"][:, :, var_idx]) ** 2).mean().item()
        print(f"    {name:>8}: direct_mse={direct_var_mse:.4f} residual_mse={residual_var_mse:.4f} combined_mse={pred_var_mse:.4f}")

    # Save
    results["correlations"] = correlations
    out_path = Path(__file__).resolve().parent / "error_analysis_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
