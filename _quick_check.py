import sys; sys.path.insert(0, '.')
from tsf_data import DataConfig, build_data_bundle
config = DataConfig(data_path='datasets/ETT-small/ETTh1.csv', seq_len=96, pred_len=96, features='M', split_points=(8640,11520))
bundle = build_data_bundle(config)
print(f'Variables: {len(bundle.input_columns)}')
print(f'Train: {len(bundle.datasets["train"])}, Val: {len(bundle.datasets["val"])}, Test: {len(bundle.datasets["test"])}')
sample = bundle.datasets['train'][0]
print(f'x shape: {sample["x"].shape}, y shape: {sample["y"].shape}')

from hypertsf_layers import HyperbolicTSF
import torch
model = HyperbolicTSF(input_length=96, pred_length=96, num_variables=7, tangent_dim=32, hidden_dim=64, manifold='poincare')
x = torch.randn(2, 96, 7)
out = model(x)
print(f'Model output shape: {out.shape}')
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
print('All checks passed!')
