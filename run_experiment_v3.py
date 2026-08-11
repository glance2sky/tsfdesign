"""Standard TSF benchmark for HyperbolicTSF with v1-compatible defaults.

This experiment uses the model's default configuration:
  - spatial_rank=None (full projection, no low-rank)
  - temporal_rank=None (full Linear, no low-rank)
  - hgcn_residual_init=None (no learnable residual)
  - use_time_identity=False (no time position encoding)
  - LocalTrendResidual active but trend_scale starts at 0

This is effectively v1 + zero-initialized trend residual + diagnostics.
The two error-driven forecast-head extensions below are opt-in.
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


# ---------------------------------------------------------------------------
# experiment config
# ---------------------------------------------------------------------------
DATASET = "ETTh1"
DATA_PATH = "datasets/ETT-small/ETTh1.csv"
SEQ_LEN = 96
PRED_LENGTHS = [96, 192, 336, 720]
SPLIT_POINTS = (8640, 11520)

# model hyper-params (use defaults)
TANGENT_DIM = 32
HIDDEN_DIM = 64
MANIFOLD = "poincare"
DROPOUT = 0.1
USE_REVIN = True
USE_LINEAR_RESIDUAL = True
USE_MULTISCALE_PROJECTION = False
USE_ADAPTIVE_PATH_FUSION = False
USE_PATH_AMPLITUDE_CALIBRATION = False
USE_OUTPUT_MULTISCALE_RESIDUAL = False
USE_FREQUENCY_RESIDUAL = False
OUTPUT_MULTISCALE_FACTORS = (2, 4, 8)
FREQUENCY_HARMONICS = 8
USE_TREND_DIFFERENCE_RESIDUAL = False
TREND_DIFFERENCE_WINDOWS = (12, 24, 48, 96)
TREND_DIFFERENCE_MAX_AMPLITUDE = 0.25
USE_EXPLICIT_PERIODIC_RESIDUAL = False
EXPLICIT_PERIODS = (12, 24, 48)
EXPLICIT_PERIODIC_MAX_AMPLITUDE = 0.25
USE_VARIABLE_HIERARCHY = False
VARIABLE_HIERARCHY_GROUPS = 3
USE_TEMPORAL_HIERARCHY = False
TEMPORAL_HIERARCHY_FACTORS = (2, 4, 8)
USE_RECURSIVE_TEMPORAL_HIERARCHY = False
RECURSIVE_TEMPORAL_FACTORS = (2, 2, 2)
USE_PATCH_TOKENS = False
PATCH_LENGTHS = (8, 16, 32)
PATCH_STRIDES = (4, 8, 16)
PATCH_HIDDEN_DIM = 64

# training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
RESULT_FILENAME = "experiment_results_v3.json"


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
def evaluate(model: HyperbolicTSF, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    preds, trues = [], []
    diag_accum = {}
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x, return_aux=True)
        preds.append(out["prediction"].cpu())
        trues.append(y.cpu())
        
        # collect diagnostics
        enc = out["encoder"]
        head = out["head"]
        scalar_keys = [
            "spatial_graph_entropy", "temporal_graph_entropy",
            "variable_weight_entropy", "fusion_gate_mean", "fusion_gate_std",
            "spatial_graph_mix", "temporal_graph_mix",
            "spatial_tangent_norm", "temporal_tangent_norm",
            "spatial_prior_dynamic_gap", "temporal_prior_dynamic_gap",
            "spatial_curvature", "temporal_curvature",
            "assignment_entropy", "group_graph_entropy",
            "group_graph_mix", "hierarchy_mix",
            "hierarchy_contribution", "leaf_tangent_norm",
            "group_tangent_norm", "global_tangent_norm",
            "temporal_hierarchy_mix",
            "temporal_hierarchy_contribution",
            "temporal_hierarchy_fine_norm",
            "temporal_hierarchy_global_norm",
            "recursive_temporal_hierarchy_mix",
            "recursive_temporal_hierarchy_contribution",
            "recursive_temporal_level_contribution_mean",
            "recursive_temporal_level_contribution_std",
            "recursive_temporal_level_graph_entropy_mean",
            "recursive_temporal_level_graph_entropy_std",
            "recursive_temporal_level_graph_mix_mean",
            "recursive_temporal_hierarchy_fine_norm",
            "recursive_temporal_hierarchy_global_norm",
            "recursive_temporal_hierarchy_depth",
            "patch_scale_gate_mean",
            "patch_scale_gate_std",
            "patch_local_contribution",
            "patch_token_contribution",
            "patch_correction_abs_mean",
            "patch_token_entropy",
        ]
        for k in scalar_keys:
            if k in enc:
                diag_accum.setdefault(k, []).append(float(enc[k].item()))
        
        head_keys = [
            "trend_scale", "direct_abs_mean", "residual_abs_mean",
            "residual_to_direct_ratio",
            "scale_gate_mean", "scale_gate_std",
            "direct_weight_mean", "residual_weight_mean",
            "path_weight_std", "adaptive_correction_abs_mean",
            "scale_contribution_mean", "scale_contribution_std",
            "direct_scale_mean", "residual_scale_mean",
            "calibration_scale_std", "calibration_correction_abs_mean",
            "calibrated_direct_abs_mean", "calibrated_residual_abs_mean",
            "calibrated_residual_to_direct_ratio",
            "output_multiscale_gate_mean",
            "output_multiscale_contribution_mean",
            "frequency_gate",
            "frequency_contribution_mean",
            "output_residual_abs_mean",
            "trend_difference_amplitude_abs_mean",
            "trend_difference_contribution_mean",
            "explicit_periodic_amplitude_abs_mean",
            "explicit_periodic_contribution_mean",
        ]
        for k in head_keys:
            if k in head:
                diag_accum.setdefault(k, []).append(float(head[k].item()))

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    mse = float(((preds - trues) ** 2).mean())
    mae = float((preds - trues).abs().mean())

    result = {"mse": mse, "mae": mae}
    for k, vals in diag_accum.items():
        result[k] = float(np.mean(vals))
    return result


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

    # use default configuration
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
        use_multiscale_projection=USE_MULTISCALE_PROJECTION,
        use_adaptive_path_fusion=USE_ADAPTIVE_PATH_FUSION,
        use_path_amplitude_calibration=USE_PATH_AMPLITUDE_CALIBRATION,
        use_output_multiscale_residual=USE_OUTPUT_MULTISCALE_RESIDUAL,
        output_multiscale_factors=OUTPUT_MULTISCALE_FACTORS,
        use_frequency_residual=USE_FREQUENCY_RESIDUAL,
        frequency_harmonics=FREQUENCY_HARMONICS,
        use_trend_difference_residual=USE_TREND_DIFFERENCE_RESIDUAL,
        trend_difference_windows=TREND_DIFFERENCE_WINDOWS,
        trend_difference_max_amplitude=TREND_DIFFERENCE_MAX_AMPLITUDE,
        use_explicit_periodic_residual=USE_EXPLICIT_PERIODIC_RESIDUAL,
        explicit_periods=EXPLICIT_PERIODS,
        explicit_periodic_max_amplitude=EXPLICIT_PERIODIC_MAX_AMPLITUDE,
        use_variable_hierarchy=USE_VARIABLE_HIERARCHY,
        variable_hierarchy_groups=VARIABLE_HIERARCHY_GROUPS,
        use_temporal_hierarchy=USE_TEMPORAL_HIERARCHY,
        temporal_hierarchy_factors=TEMPORAL_HIERARCHY_FACTORS,
        use_recursive_temporal_hierarchy=USE_RECURSIVE_TEMPORAL_HIERARCHY,
        recursive_temporal_factors=RECURSIVE_TEMPORAL_FACTORS,
        use_patch_tokens=USE_PATCH_TOKENS,
        patch_lengths=PATCH_LENGTHS,
        patch_strides=PATCH_STRIDES,
        patch_hidden_dim=PATCH_HIDDEN_DIM,
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
    epoch_logs = []

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
        val_result = evaluate(model, val_loader, device)
        val_mse = val_result["mse"]
        val_mae = val_result["mae"]
        scheduler.step(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]

        # diagnostic info
        diag_str = ""
        if epoch == 1 or epoch == EPOCHS or val_mse <= best_val_loss:
            diag_str = (
                f" | graph_ent(s/t)={val_result.get('spatial_graph_entropy', 0):.3f}/"
                f"{val_result.get('temporal_graph_entropy', 0):.3f}"
                f" gate={val_result.get('fusion_gate_mean', 0):.3f}"
                f" trend={val_result.get('trend_scale', 0):.4f}"
                f" res/dir={val_result.get('residual_to_direct_ratio', 0):.3f}"
            )

        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | "
            f"train={avg_train:.4f} | val_mse={val_mse:.4f} mae={val_mae:.4f} | "
            f"lr={current_lr:.2e}{diag_str}"
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(avg_train, 4),
            "val_mse": round(val_mse, 4),
            "val_mae": round(val_mae, 4),
            "lr": current_lr,
            **{k: round(v, 4) for k, v in val_result.items() if k not in ("mse", "mae")},
        })

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
    test_result = evaluate(model, test_loader, device)
    test_mse = test_result["mse"]
    test_mae = test_result["mae"]
    print(f"  >>> TEST  MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    # final diagnostics
    print(f"  Diagnostics:")
    print(f"    graph_entropy(s/t) = {test_result.get('spatial_graph_entropy', 0):.4f} / {test_result.get('temporal_graph_entropy', 0):.4f}")
    print(f"    var_weight_entropy = {test_result.get('variable_weight_entropy', 0):.4f}")
    print(f"    fusion_gate        = {test_result.get('fusion_gate_mean', 0):.4f} ± {test_result.get('fusion_gate_std', 0):.4f}")
    print(f"    graph_mix(s/t)     = {test_result.get('spatial_graph_mix', 0):.4f} / {test_result.get('temporal_graph_mix', 0):.4f}")
    print(f"    tangent_norm(s/t)  = {test_result.get('spatial_tangent_norm', 0):.4f} / {test_result.get('temporal_tangent_norm', 0):.4f}")
    print(f"    prior_dyn_gap(s/t) = {test_result.get('spatial_prior_dynamic_gap', 0):.4f} / {test_result.get('temporal_prior_dynamic_gap', 0):.4f}")
    print(f"    curvature(s/t)     = {test_result.get('spatial_curvature', 0):.4f} / {test_result.get('temporal_curvature', 0):.4f}")
    print(f"    trend_scale        = {test_result.get('trend_scale', 0):.4f}")
    print(f"    direct_abs_mean    = {test_result.get('direct_abs_mean', 0):.4f}")
    print(f"    residual_abs_mean  = {test_result.get('residual_abs_mean', 0):.4f}")
    print(f"    residual/direct    = {test_result.get('residual_to_direct_ratio', 0):.4f}")
    print(f"    scale_gate         = {test_result.get('scale_gate_mean', 0):.4f} ± "
          f"{test_result.get('scale_gate_std', 0):.4f}")
    print(f"    path_weights       = {test_result.get('direct_weight_mean', 0):.4f} / "
          f"{test_result.get('residual_weight_mean', 0):.4f}")
    print(f"    path_weight_std    = {test_result.get('path_weight_std', 0):.4f}")

    return {
        "pred_len": pred_len,
        "test_mse": round(test_mse, 4),
        "test_mae": round(test_mae, 4),
        "best_val_mse": round(best_val_loss, 4),
        "train_time_s": round(elapsed, 1),
        "total_params": total_params,
        "final_diagnostics": {k: round(v, 4) for k, v in test_result.items() if k not in ("mse", "mae")},
        "epoch_logs": epoch_logs,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET}  ({DATA_PATH})")
    print(f"seq_len={SEQ_LEN}, pred_len={PRED_LENGTHS}")
    print(f"model: manifold={MANIFOLD}, tangent_dim={TANGENT_DIM}, hidden_dim={HIDDEN_DIM}")
    print(f"       configuration: DEFAULT (v1-compatible, no low-rank, no time_identity)")
    print(f"       multiscale_projection={USE_MULTISCALE_PROJECTION}, "
          f"adaptive_path_fusion={USE_ADAPTIVE_PATH_FUSION}, "
          f"path_amplitude_calibration={USE_PATH_AMPLITUDE_CALIBRATION}, "
          f"output_multiscale_residual={USE_OUTPUT_MULTISCALE_RESIDUAL}, "
          f"frequency_residual={USE_FREQUENCY_RESIDUAL}, "
          f"trend_difference_residual={USE_TREND_DIFFERENCE_RESIDUAL}, "
          f"explicit_periodic_residual={USE_EXPLICIT_PERIODIC_RESIDUAL}, "
          f"variable_hierarchy={USE_VARIABLE_HIERARCHY} "
          f"(groups={VARIABLE_HIERARCHY_GROUPS}), "
          f"temporal_hierarchy={USE_TEMPORAL_HIERARCHY} "
          f"(factors={TEMPORAL_HIERARCHY_FACTORS}), "
          f"recursive_temporal_hierarchy="
          f"{USE_RECURSIVE_TEMPORAL_HIERARCHY} "
          f"(factors={RECURSIVE_TEMPORAL_FACTORS})")
    print(
        f"       patch_tokens={USE_PATCH_TOKENS} "
        f"(lengths={PATCH_LENGTHS}, strides={PATCH_STRIDES})"
    )
    variant = "v3-default"
    if USE_MULTISCALE_PROJECTION and USE_ADAPTIVE_PATH_FUSION:
        variant = "v4-multiscale-adaptive"
    elif USE_MULTISCALE_PROJECTION:
        variant = "v4-multiscale"
    elif USE_ADAPTIVE_PATH_FUSION:
        variant = "v4-adaptive"
    if USE_PATH_AMPLITUDE_CALIBRATION:
        variant = f"{variant}-calibrated"
    if USE_OUTPUT_MULTISCALE_RESIDUAL:
        variant = f"{variant}-output-ms"
    if USE_FREQUENCY_RESIDUAL:
        variant = f"{variant}-frequency"
    if USE_TREND_DIFFERENCE_RESIDUAL:
        variant = f"{variant}-trend-difference"
    if USE_EXPLICIT_PERIODIC_RESIDUAL:
        variant = f"{variant}-explicit-periodic"
    if USE_VARIABLE_HIERARCHY:
        variant = f"{variant}-variable-hierarchy"
    if USE_TEMPORAL_HIERARCHY:
        variant = f"{variant}-temporal-hierarchy"
    if USE_RECURSIVE_TEMPORAL_HIERARCHY:
        variant = f"{variant}-recursive-temporal-hierarchy"
    if USE_PATCH_TOKENS:
        variant = f"{variant}-patch-tokens"

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
    print(f"\n\n{'='*80}")
    print(f"  RESULTS SUMMARY (v3: default config)  -  {DATASET}  (seq_len={SEQ_LEN})")
    print(f"{'='*80}")
    print(f"  {'pred_len':>8}  {'MSE':>8}  {'MAE':>8}  {'val_MSE':>8}  {'time(s)':>8}  {'params':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in results:
        print(f"  {r['pred_len']:>8}  {r['test_mse']:>8.4f}  {r['test_mae']:>8.4f}  "
              f"{r['best_val_mse']:>8.4f}  {r['train_time_s']:>8.1f}  {r['total_params']:>10,}")

    # --- comparison with v1 and v2 ---
    v1 = {96: (0.4675, 0.4484), 192: (0.5231, 0.4799), 
          336: (0.5614, 0.5033), 720: (0.6870, 0.5802)}
    v2 = {96: (0.4820, 0.4638), 192: (0.5378, 0.4941),
          336: (0.6021, 0.5301), 720: (0.7163, 0.6008)}
    
    print(f"\n  {'pred_len':>8}  {'v1 MSE':>8}  {'v3 MSE':>8}  {'v1→v3':>8}  {'v2 MSE':>8}  {'v2→v3':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        pl = r["pred_len"]
        if pl in v1 and pl in v2:
            v1_mse, _ = v1[pl]
            v2_mse, _ = v2[pl]
            v3_mse = r["test_mse"]
            v1_change = f"{(v3_mse - v1_mse) / v1_mse * 100:+.1f}%"
            v2_change = f"{(v3_mse - v2_mse) / v2_mse * 100:+.1f}%"
            print(f"  {pl:>8}  {v1_mse:>8.4f}  {v3_mse:>8.4f}  {v1_change:>8}  "
                  f"{v2_mse:>8.4f}  {v2_change:>8}")

    # save results
    out_path = Path(__file__).resolve().parent / RESULT_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET,
                "seq_len": SEQ_LEN,
                "model": f"HyperbolicTSF-{variant}",
                "manifold": MANIFOLD,
                "tangent_dim": TANGENT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "configuration": variant,
                "spatial_rank": None,
                "temporal_rank": None,
                "hgcn_residual_init": None,
                "use_time_identity": False,
                "use_multiscale_projection": USE_MULTISCALE_PROJECTION,
                "use_adaptive_path_fusion": USE_ADAPTIVE_PATH_FUSION,
                "use_path_amplitude_calibration": USE_PATH_AMPLITUDE_CALIBRATION,
                "use_output_multiscale_residual": USE_OUTPUT_MULTISCALE_RESIDUAL,
                "output_multiscale_factors": OUTPUT_MULTISCALE_FACTORS,
                "use_frequency_residual": USE_FREQUENCY_RESIDUAL,
                "frequency_harmonics": FREQUENCY_HARMONICS,
                "use_trend_difference_residual": (
                    USE_TREND_DIFFERENCE_RESIDUAL
                ),
                "trend_difference_windows": TREND_DIFFERENCE_WINDOWS,
                "trend_difference_max_amplitude": (
                    TREND_DIFFERENCE_MAX_AMPLITUDE
                ),
                "use_explicit_periodic_residual": (
                    USE_EXPLICIT_PERIODIC_RESIDUAL
                ),
                "explicit_periods": EXPLICIT_PERIODS,
                "explicit_periodic_max_amplitude": (
                    EXPLICIT_PERIODIC_MAX_AMPLITUDE
                ),
                "use_variable_hierarchy": USE_VARIABLE_HIERARCHY,
                "variable_hierarchy_groups": VARIABLE_HIERARCHY_GROUPS,
                "use_temporal_hierarchy": USE_TEMPORAL_HIERARCHY,
                "temporal_hierarchy_factors": TEMPORAL_HIERARCHY_FACTORS,
                "use_recursive_temporal_hierarchy": (
                    USE_RECURSIVE_TEMPORAL_HIERARCHY
                ),
                "recursive_temporal_factors": RECURSIVE_TEMPORAL_FACTORS,
                "use_patch_tokens": USE_PATCH_TOKENS,
                "patch_lengths": PATCH_LENGTHS,
                "patch_strides": PATCH_STRIDES,
                "patch_hidden_dim": PATCH_HIDDEN_DIM,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
