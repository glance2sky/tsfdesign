# Hyperbolic TSF Model

The model is organized as follows:

```text
RevIN
  -> variable-preserving manifold embedding
  -> parallel temporal/variable HNN + HGCN encoding
```

## Layer 1: RevIN

The input is:

```text
[batch, time, variables]
```

RevIN computes per-sample, per-variable temporal statistics and returns:

```text
x_norm:   [B, L, C]
location: [B, 1, C]
scale:    [B, 1, C]
```

The inverse operation is available for future prediction heads:

```text
y_hat_norm -> y_hat
```

This is separate from the train-only dataset standardizer. The data
standardizer aligns global training scale, while RevIN handles local
window-level distribution shift.

## Layer 2: Manifold Embedding

The input is:

```text
[batch, time, variables]
```

Each scalar variable value is projected to a shared tangent dimension. A
learned variable identity is added before mapping from the origin to the
selected Euclidean, Poincare-ball, or Lorentz manifold:

```text
[B, L, C] -> [B, L, C, tangent_dim] -> [B, L, C, manifold_dim]
```

The variable axis is intentionally preserved. A single linear layer from
`C` to `d` would mix variables too early and make the later variable graph
less interpretable.

## Layer 3: HAO-Style Dual Graph Encoding

The current main branch intentionally does not use PatchTST-style patching.
The input window is decomposed in two parallel ways, following the central
spatial/temporal construction in HAO:

```text
variable branch: [B, L, C, d] -> [B, C, L * d]
temporal branch: [B, L, C, d] -> [B, L, C * d]
```

The variable branch treats each variable as a graph node and the temporal
branch treats each time point as a graph node. Each branch has:

1. a learnable symmetric prior graph;
2. an HNN-style tangent projection and manifold mapping;
3. a dense-batch HGCN block with tangent-space graph aggregation;
4. a dynamic graph estimated from pairwise manifold distances;
5. a differentiable blend of the prior and dynamic graphs.

The layer returns both graphs, their distance matrices, node features, and a
cross-branch interaction tensor:

```text
spatial_tangent:  [B, C, H]
temporal_tangent: [B, L, H]
interaction:      [B, C, L]
```

## Forecasting Head

The end-to-end model is:

```text
[B,L,C]
  -> RevIN
  -> manifold embedding
  -> parallel temporal/variable HNN + HGCN
  -> interaction-guided variable context
  -> direct multi-horizon head
  -> RevIN inverse
  -> [B,pred_len,C_out]
```

The interaction tensor is normalized over variables for each time point.
Therefore, every temporal node receives a different weighted combination of
variable-node representations. A sigmoid gate controls how much this
variable context changes the temporal representation.

The forecasting head contains two paths:

1. a graph-aware direct projection from the historical temporal context to all
   future horizons;
2. an optional linear residual path from the normalized input, which preserves
   simple trend and seasonal extrapolation.

The two paths are added before inverse RevIN. `target_indices` supports `S`,
`M`, and `MS` forecasting modes without requiring the prediction head to emit
unused variables.

## Structural Stabilization

The current implementation adds three constraints motivated by the ETTh1
training behavior:

1. The variable-branch HNN projection can use a low-rank factorization. This
   avoids a fully free mapping from every historical position to every
   variable node, reducing memorization of the fixed input window.
2. HGCN uses a learnable residual coefficient. The second graph aggregation
   therefore cannot arbitrarily erase the original node representation.
3. The direct head uses shared low-rank temporal bases and a local
   level/slope residual. This gives long horizons a shared temporal structure
   instead of a separate unconstrained parameter for every history/future
   position pair.

The model exposes diagnostics through `forward(return_aux=True)`:

```text
spatial_graph_entropy       normalized variable-graph entropy
temporal_graph_entropy      normalized time-graph entropy
variable_weight_entropy     normalized variable-to-time attention entropy
spatial_graph_mix           dynamic/prior graph mixing coefficient
temporal_graph_mix          dynamic/prior graph mixing coefficient
spatial_prior_dynamic_gap   distance between prior and dynamic graphs
temporal_prior_dynamic_gap  distance between prior and dynamic graphs
spatial_tangent_norm        mean spatial tangent norm
temporal_tangent_norm       mean temporal tangent norm
fusion_gate_mean            mean variable-context gate
fusion_gate_std             gate diversity
```

These quantities are intended for logging during training. Very low graph
entropy indicates near one-hot or collapsed neighborhoods; entropy close to
one indicates a nearly uniform graph. Rapidly increasing tangent norms may
indicate manifold instability. A fusion gate near zero means the variable
graph is being ignored, while a gate with almost zero standard deviation
indicates that all time points receive nearly identical variable context.

## Optional Error-Driven Head Extensions

The forecasting head provides two opt-in extensions motivated by the ETTh1
error analysis:

1. `use_multiscale_projection=True` adds pooled temporal projections at
   factors such as `(1, 2, 4)`. The fine-scale branch remains the original
   projection, while coarse branches use zero-initialized bounded gates.
2. `use_adaptive_path_fusion=True` predicts per-variable and per-horizon
   coefficients for the graph-aware direct path and the normalized-input
   residual path. Both coefficients start at one, preserving the original
   `direct + residual` behavior at initialization.

The additional head diagnostics are:

```text
scale_gate_mean
scale_gate_std
direct_weight_mean
residual_weight_mean
path_weight_std
adaptive_correction_abs_mean
path_weights
```

These options are disabled by default so the v3 baseline remains directly
reproducible. They should be evaluated separately and together rather than
interpreted as a single undifferentiated architectural change.

## Capacity-Preserving Defaults

The public `HyperbolicTSF` defaults preserve the original baseline capacity:

```text
spatial_rank=None
temporal_rank=None
hgcn_residual_init=None
use_time_identity=False
```

The optional local trend branch is zero-initialized and bounded by a
`tanh`-parameterized scale. It therefore cannot dominate the graph-aware
forecast at initialization. The head exposes `trend_scale`,
`direct_abs_mean`, `residual_abs_mean`, and
`residual_to_direct_ratio` through `return_aux=True`, making it possible to
detect whether the residual path is becoming a shortcut.
