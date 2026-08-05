from pathlib import Path

import numpy as np
import pandas as pd

from tsf_data import DataConfig, build_data_bundle


def _write_fixture(path: Path) -> None:
    size = 20
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size, freq="h"),
            "x": np.arange(size, dtype=float),
            "y": np.arange(size, dtype=float) * 10.0,
        }
    )
    frame.to_csv(path, index=False)


def test_train_only_scaling_and_split_windows(tmp_path: Path) -> None:
    path = tmp_path / "toy.csv"
    _write_fixture(path)
    config = DataConfig(
        data_path=path,
        seq_len=3,
        pred_len=2,
        label_len=2,
        features="MS",
        target="y",
        train_ratio=0.6,
        val_ratio=0.2,
        split_points=(12, 16),
        add_time_features=True,
    )
    bundle = build_data_bundle(config)

    assert bundle.split_points == {"train": (0, 12), "val": (12, 16), "test": (16, 20)}
    assert bundle.input_dim == 2
    assert bundle.output_dim == 1
    assert len(bundle.datasets["train"]) == 8
    assert len(bundle.datasets["val"]) == 3
    assert len(bundle.datasets["test"]) == 3
    sample = bundle.datasets["test"][0]
    assert tuple(sample["x"].shape) == (3, 2)
    assert tuple(sample["y"].shape) == (2, 1)
    assert tuple(sample["y_context"].shape) == (2, 1)
    assert tuple(sample["decoder_y"].shape) == (4, 1)
    assert tuple(sample["decoder_mark"].shape) == (4, 10)


def test_validation_and_test_targets_do_not_cross_split_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "toy.csv"
    _write_fixture(path)
    config = DataConfig(
        data_path=path,
        seq_len=3,
        pred_len=2,
        features="M",
        train_ratio=0.6,
        val_ratio=0.2,
    )
    bundle = build_data_bundle(config)

    for name, (_, split_end) in bundle.split_points.items():
        dataset = bundle.datasets[name]
        split_start, _ = bundle.split_points[name]
        for index in range(len(dataset)):
            sample = dataset[index]
            start = int(sample["start_idx"])
            target_start = int(sample["target_start_idx"])
            target_end = int(sample["target_end_idx"])
            assert target_start >= split_start
            assert target_end <= split_end
            assert start + config.seq_len == target_start


def test_inverse_target_restores_scaled_values(tmp_path: Path) -> None:
    path = tmp_path / "toy.csv"
    _write_fixture(path)
    config = DataConfig(
        data_path=path,
        seq_len=3,
        pred_len=2,
        features="MS",
        target="y",
        train_ratio=0.6,
        val_ratio=0.2,
    )
    bundle = build_data_bundle(config)
    sample = bundle.datasets["test"][0]
    restored = bundle.inverse_target(sample["y"].numpy()[None, ...])
    np.testing.assert_allclose(restored[0, :, 0], [160.0, 170.0], rtol=0, atol=1e-4)


def test_label_len_context_uses_observed_history_only(tmp_path: Path) -> None:
    path = tmp_path / "toy.csv"
    _write_fixture(path)
    config = DataConfig(
        data_path=path,
        seq_len=4,
        pred_len=2,
        label_len=3,
        features="MS",
        target="y",
        train_ratio=0.6,
        val_ratio=0.2,
    )
    bundle = build_data_bundle(config)
    sample = bundle.datasets["test"][0]
    context = bundle.inverse_target(sample["y_context"].numpy()[None, ...])
    future = bundle.inverse_target(sample["y"].numpy()[None, ...])
    np.testing.assert_allclose(context[0, :, 0], [130.0, 140.0, 150.0], atol=1e-4)
    np.testing.assert_allclose(future[0, :, 0], [160.0, 170.0], atol=1e-4)
