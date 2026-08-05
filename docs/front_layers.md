# Hyperbolic TSF Model

The first three conceptual layers are organized as follows:

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

The fourth layer is intentionally not implemented.

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
