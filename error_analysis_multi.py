"""Multi-horizon error analysis: compare error patterns across pred_len=96/192/336."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
SPLIT_POINTS = (8640, 11520)
TANGENT_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train_best(pred_len, num_variables, train_loader, val_loader):
    model = HyperbolicTSF(
        input_length=SEQ_LEN, pred_length=pred_len,
        num_variables=num_variables, tangent_dim=TANGENT_DIM,
        hidden_dim=HIDDEN_DIM, manifold="poincare", dropout=0.1,
        use_revin=True, use_linear_residual=True,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for batch in val_loader:
                x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
                vp.append(model(x).cpu()); vt.append(y.cpu())
            val_mse = float(((torch.cat(vp) - torch.cat(vt)) ** 2).mean())
        scheduler.step(val_mse)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stop epoch {epoch}")
                break
        print(f"    Epoch {epoch:02d} | val_mse={val_mse:.4f}")
    model.load_state_dict(best_state)
    return model.to(DEVICE)


@torch.no_grad()
def analyze_horizon(model, test_loader, pred_len, variable_names):
    model.eval()
    preds, trues, inputs = [], [], []
    direct_forecasts, residual_forecasts = [], []
    start_indices = []
    for batch in test_loader:
        x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
        out = model(x, return_aux=True)
        preds.append(out["prediction"].cpu())
        trues.append(y.cpu())
        inputs.append(x.cpu())
        components = decompose_head_forecasts(
            model,
            out["head"]["direct"],
            out["head"]["residual"],
            out["revin_state"],
        )
        direct_forecasts.append(components["direct_forecast"].cpu())
        residual_forecasts.append(components["residual_forecast"].cpu())
        start_indices.append(batch["start_idx"].cpu())
    preds = torch.cat(preds); trues = torch.cat(trues); inputs = torch.cat(inputs)
    directs = torch.cat(direct_forecasts)
    residuals = torch.cat(residual_forecasts)
    start_indices = torch.cat(start_indices).numpy()

    errors = (preds - trues) ** 2
    n = preds.size(0)

    # Per-variable MSE
    var_mse = errors.mean(dim=(0, 1)).tolist()

    # Horizon growth
    horizon_mse = errors.mean(dim=(0, 2)).tolist()
    growth = horizon_mse[-1] / max(horizon_mse[0], 1e-8)

    # Per-variable direct vs residual
    var_direct_mse = ((directs - trues) ** 2).mean(dim=(0, 1)).tolist()
    var_residual_mse = ((residuals - trues) ** 2).mean(dim=(0, 1)).tolist()

    # Which path is better per variable
    direct_better = [d < r for d, r in zip(var_direct_mse, var_residual_mse)]

    # Input characteristics correlation
    sample_mse = errors.mean(dim=(1, 2)).numpy()
    input_vol = inputs.std(dim=1).mean(dim=1).numpy()
    half = inputs.size(1) // 2
    input_trend = (inputs[:, half:, :].mean(dim=1) - inputs[:, :half, :].mean(dim=1)).mean(dim=1).numpy()
    input_range = (inputs.max(dim=1).values - inputs.min(dim=1).values).mean(dim=1).numpy()

    # Pearson correlations
    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a**2).sum() * (b**2).sum())
        return float((a * b).sum() / d) if d > 1e-12 else 0.0

    # Temporal clustering: check if worst samples are clustered
    worst_indices = sample_mse.argsort()[-50:][::-1].tolist()
    best_indices = sample_mse.argsort()[:50].tolist()

    # Gap analysis for worst samples
    worst_starts = np.sort(start_indices[worst_indices])
    best_starts = np.sort(start_indices[best_indices])
    worst_gaps = np.diff(worst_starts)
    best_gaps = np.diff(best_starts)
    median_worst_gap = worst_gaps[len(worst_gaps)//2]
    median_best_gap = best_gaps[len(best_gaps)//2]

    # Hour-of-day analysis (ETTh1 is hourly)
    # Group by the actual forecast-start timestamp, not test-loader position.
    frame_timestamps = pd.to_datetime(
        getattr(test_loader.dataset, "time_values", None)
    )
    forecast_indices = start_indices + SEQ_LEN
    forecast_timestamps = frame_timestamps[forecast_indices]
    hours_of_day = forecast_timestamps.hour.to_numpy()
    hour_errors = {}
    for i, h in enumerate(hours_of_day):
        hour_errors.setdefault(h, []).append(float(sample_mse[i]))
    hour_avg = {h: float(np.mean(v)) for h, v in hour_errors.items()}

    # Day-of-week (168 hours per week)
    dows = forecast_timestamps.dayofweek.to_numpy()
    dow_errors = {}
    for i, d in enumerate(dows):
        dow_errors.setdefault(d, []).append(float(sample_mse[i]))
    dow_avg = {d: float(np.mean(v)) for d, v in dow_errors.items()}

    # Worst vs best input stats
    worst_sample_mse_avg = float(np.mean([sample_mse[i] for i in worst_indices]))
    best_sample_mse_avg = float(np.mean([sample_mse[i] for i in best_indices]))
    worst_vol_avg = float(np.mean([input_vol[i] for i in worst_indices]))
    best_vol_avg = float(np.mean([input_vol[i] for i in best_indices]))
    worst_trend_avg = float(np.mean([input_trend[i] for i in worst_indices]))
    best_trend_avg = float(np.mean([input_trend[i] for i in best_indices]))
    worst_range_avg = float(np.mean([input_range[i] for i in worst_indices]))
    best_range_avg = float(np.mean([input_range[i] for i in best_indices]))

    return {
        "pred_len": pred_len,
        "n_samples": n,
        "overall_mse": float(sample_mse.mean()),
        "variable_mse": dict(zip(variable_names, [round(v, 4) for v in var_mse])),
        "variable_direct_mse": dict(zip(variable_names, [round(v, 4) for v in var_direct_mse])),
        "variable_residual_mse": dict(zip(variable_names, [round(v, 4) for v in var_residual_mse])),
        "direct_better_per_var": dict(zip(variable_names, direct_better)),
        "horizon_growth": round(growth, 2),
        "first_24_mse": round(float(np.mean(horizon_mse[:24])), 4),
        "last_24_mse": round(float(np.mean(horizon_mse[-24:])), 4),
        "error_vs_volatility": round(pearson(sample_mse, input_vol), 4),
        "error_vs_trend": round(pearson(sample_mse, input_trend), 4),
        "error_vs_range": round(pearson(sample_mse, input_range), 4),
        "median_worst_gap": median_worst_gap,
        "median_best_gap": median_best_gap,
        "worst_vs_best": {
            "mse": (round(worst_sample_mse_avg, 4), round(best_sample_mse_avg, 4)),
            "volatility": (round(worst_vol_avg, 4), round(best_vol_avg, 4)),
            "trend": (round(worst_trend_avg, 4), round(best_trend_avg, 4)),
            "range": (round(worst_range_avg, 4), round(best_range_avg, 4)),
        },
        "hour_mse": {str(h): round(v, 4) for h, v in sorted(hour_avg.items())},
        "dow_mse": {str(d): round(v, 4) for d, v in sorted(dow_avg.items())},
    }


def main():
    print(f"=== Multi-Horizon Error Analysis ===\n")
    
    all_results = {}
    for pred_len in [96, 192, 336]:
        print(f"\n--- pred_len={pred_len} ---")
        config = DataConfig(
            data_path=DATA_PATH, seq_len=SEQ_LEN, pred_len=pred_len,
            features="M", split_points=SPLIT_POINTS, scaler="standard",
        )
        bundle = build_data_bundle(config)
        num_variables = len(bundle.input_columns)
        variable_names = bundle.input_columns

        train_loader = DataLoader(bundle.datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate, drop_last=True)
        val_loader = DataLoader(bundle.datasets["val"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)
        test_loader = DataLoader(bundle.datasets["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)

        model = train_best(pred_len, num_variables, train_loader, val_loader)
        result = analyze_horizon(model, test_loader, pred_len, variable_names)
        all_results[pred_len] = result
        del model
        torch.cuda.empty_cache()

    # ---- Print summary ----
    print(f"\n\n{'='*80}")
    print(f"  MULTI-HORIZON ERROR ANALYSIS SUMMARY")
    print(f"{'='*80}")

    for pred_len, r in all_results.items():
        print(f"\n  === pred_len={pred_len} ===")
        print(f"  Overall MSE: {r['overall_mse']:.4f}")
        print(f"  Horizon growth: {r['horizon_growth']}x  (first24={r['first_24_mse']:.4f}, last24={r['last_24_mse']:.4f})")
        print(f"  Correlations: vol={r['error_vs_volatility']:+.4f}, trend={r['error_vs_trend']:+.4f}, range={r['error_vs_range']:+.4f}")
        print(f"  Temporal clustering: worst_gap={r['median_worst_gap']}, best_gap={r['median_best_gap']}")
        print(f"  Worst vs Best:")
        for metric, (w, b) in r['worst_vs_best'].items():
            print(f"    {metric:>12}: worst={w:.4f}  best={b:.4f}")
        print(f"  Variable MSE:")
        for name, mse in sorted(r['variable_mse'].items(), key=lambda x: x[1], reverse=True):
            dm = r['variable_direct_mse'][name]
            rm = r['variable_residual_mse'][name]
            db = "direct" if r['direct_better_per_var'][name] else "residual"
            print(f"    {name:>8}: combined={mse:.4f}  direct={dm:.4f}  residual={rm:.4f}  better={db}")
        print(f"  Hour-of-day MSE (top 3 worst):")
        hour_sorted = sorted(r['hour_mse'].items(), key=lambda x: x[1], reverse=True)
        for h, mse in hour_sorted[:3]:
            print(f"    hour {h:>2}: {mse:.4f}")

    # Save
    out_path = Path(__file__).resolve().parent / "error_analysis_multi.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
