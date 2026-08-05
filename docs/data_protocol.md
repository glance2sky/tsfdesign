# Forecasting Data Protocol

This project uses one leakage-aware data contract for all forecasting models.

## Sample Contract

Each sample is returned as a dictionary:

- `x`: `[seq_len, input_dim]` historical values
- `y`: `[pred_len, output_dim]` future targets
- `y_context`: `[label_len, output_dim]` optional historical target context
- `decoder_y`: `[label_len + pred_len, output_dim]` decoder context plus future
- `x_mark`: optional `[seq_len, 10]` cyclical calendar features
- `y_mark`: optional `[pred_len, 10]` cyclical calendar features
- `decoder_mark`: optional `[label_len + pred_len, 10]` cyclical calendar features
- `start_idx`, `target_start_idx`, `target_end_idx`: raw-series indices

The `MS` mode uses all numeric variables as input and only the configured
target column as output. The `M` mode predicts every input variable. The `S`
mode uses one target variable for both input and output.

## Split Rules

The preferred split is an explicit chronological boundary:

```python
from tsf_data import DataConfig, build_data_bundle

config = DataConfig(
    data_path="datasets/ETT-small/ETTh1.csv",
    seq_len=96,
    label_len=48,
    pred_len=96,
    features="MS",
    target="OT",
    split_points=(8640, 11520),
    add_time_features=True,
)
bundle = build_data_bundle(config)
```

For ordinary custom datasets, `split_points` can be omitted and the default
ratio split is used. The scaler is fitted only on the training interval and
then applied to validation and test values.

Validation and test windows may use historical context immediately before
their split boundary. Their target ranges cannot cross the boundary. This
matches the causal forecasting setting used by common long-term forecasting
benchmarks.

`label_len` is optional and defaults to `0`. It exists for encoder-decoder
forecasting models that need a known historical target segment before the
future horizon. Losses should still be computed only on `y`, not on the
historical context portion of `decoder_y`.

Patchification is intentionally not part of the data protocol. Patch length,
patch stride, multi-scale patching, and patch-to-manifold mapping are model
choices, so models receive point-level windows and can patchify internally.

## Preprocessing Policy

- timestamps are parsed, sorted, and deduplicated;
- numeric columns are coerced explicitly;
- missing values raise an error by default;
- optional forward-fill mode is available for datasets where missing values
  are part of the raw format;
- scale statistics are saved through `Standardizer.state_dict()`;
- all tensors are `float32`, while scaler statistics remain `float64`.

The protocol intentionally keeps geometric embedding out of preprocessing.
Normalization happens in the data layer; Euclidean-to-hyperbolic mapping,
curvature, and manifold projection belong to the model layer.
