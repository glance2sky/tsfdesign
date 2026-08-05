"""Leakage-aware data processing for multivariate time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ScalerName = Literal["standard", "minmax", "none"]
FeatureMode = Literal["S", "M", "MS"]
MissingPolicy = Literal["raise", "ffill"]


@dataclass(frozen=True)
class DataConfig:
    """Configuration for a single regularly sampled forecasting dataset."""

    data_path: str | Path
    seq_len: int
    pred_len: int
    label_len: int = 0
    target: str | None = None
    timestamp_col: str | None = "date"
    features: FeatureMode = "M"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    split_points: tuple[int, int] | None = None
    scaler: ScalerName = "standard"
    missing_policy: MissingPolicy = "raise"
    add_time_features: bool = False
    stride: int = 1

    def __post_init__(self) -> None:
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if self.label_len < 0:
            raise ValueError("label_len must be non-negative")
        if self.label_len > self.seq_len:
            raise ValueError("label_len cannot be greater than seq_len")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.features not in {"S", "M", "MS"}:
            raise ValueError("features must be one of 'S', 'M', or 'MS'")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be in (0, 1)")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be in (0, 1)")
        if self.train_ratio + self.val_ratio >= 1 and self.split_points is None:
            raise ValueError("train_ratio + val_ratio must be less than 1")
        if self.split_points is not None:
            if len(self.split_points) != 2:
                raise ValueError("split_points must be (train_end, val_end)")
            if not 0 < self.split_points[0] < self.split_points[1]:
                raise ValueError("split_points must be strictly increasing")
        if self.features in {"S", "MS"} and not self.target:
            raise ValueError("target is required for S and MS feature modes")


class Standardizer:
    """Small serializable scaler with train-only fitting semantics."""

    def __init__(self, kind: ScalerName = "standard") -> None:
        self.kind = kind
        self.location: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        values = _as_float_array(values)
        if values.ndim != 2:
            raise ValueError("values must have shape [time, features]")
        if self.kind == "standard":
            location = np.mean(values, axis=0)
            scale = np.std(values, axis=0)
        elif self.kind == "minmax":
            location = np.min(values, axis=0)
            scale = np.max(values, axis=0) - location
        elif self.kind == "none":
            location = np.zeros(values.shape[1], dtype=np.float64)
            scale = np.ones(values.shape[1], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported scaler: {self.kind}")
        self.location = location.astype(np.float64, copy=False)
        self.scale = np.where(np.abs(scale) < 1e-12, 1.0, scale).astype(
            np.float64, copy=False
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (_as_float_array(values) - self.location) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return _as_float_array(values) * self.scale + self.location

    def state_dict(self) -> dict[str, object]:
        self._check_fitted()
        return {
            "kind": self.kind,
            "location": self.location.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "Standardizer":
        scaler = cls(str(state["kind"]))  # type: ignore[arg-type]
        scaler.location = np.asarray(state["location"], dtype=np.float64)
        scaler.scale = np.asarray(state["scale"], dtype=np.float64)
        return scaler

    def _check_fitted(self) -> None:
        if self.location is None or self.scale is None:
            raise RuntimeError("The scaler must be fitted before use")


class ForecastDataset(Dataset[dict[str, torch.Tensor]]):
    """Sliding-window dataset with explicit target timestamps."""

    def __init__(
        self,
        values: np.ndarray,
        time_values: np.ndarray,
        starts: np.ndarray,
        seq_len: int,
        pred_len: int,
        label_len: int,
        target_indices: np.ndarray,
        time_features: np.ndarray | None = None,
    ) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.time_values = np.asarray(time_values)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.time_features = (
            None
            if time_features is None
            else np.asarray(time_features, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        input_end = start + self.seq_len
        target_end = input_end + self.pred_len
        context_start = input_end - self.label_len
        y_context = self.values[context_start:input_end][:, self.target_indices]
        y_future = self.values[input_end:target_end][:, self.target_indices]
        sample = {
            "x": torch.from_numpy(self.values[start:input_end]),
            "y": torch.from_numpy(y_future),
            "y_context": torch.from_numpy(y_context),
            "decoder_y": torch.from_numpy(np.concatenate([y_context, y_future], axis=0)),
            "start_idx": torch.tensor(start, dtype=torch.long),
            "target_start_idx": torch.tensor(input_end, dtype=torch.long),
            "target_end_idx": torch.tensor(target_end, dtype=torch.long),
        }
        if self.time_features is not None:
            sample["x_mark"] = torch.from_numpy(
                self.time_features[start:input_end]
            )
            sample["y_mark"] = torch.from_numpy(
                self.time_features[input_end:target_end]
            )
            sample["y_context_mark"] = torch.from_numpy(
                self.time_features[context_start:input_end]
            )
            sample["decoder_mark"] = torch.from_numpy(
                self.time_features[context_start:target_end]
            )
        return sample


@dataclass
class DataBundle:
    """Processed arrays, split datasets, and the train-fitted scaler."""

    config: DataConfig
    frame: pd.DataFrame
    input_columns: list[str]
    target_columns: list[str]
    scaler: Standardizer
    datasets: dict[str, ForecastDataset]
    split_points: dict[str, tuple[int, int]]

    @property
    def input_dim(self) -> int:
        return len(self.input_columns)

    @property
    def output_dim(self) -> int:
        return len(self.target_columns)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Invert scaled target values, including the S/MS column selection."""
        restored = self.scaler.inverse_transform(
            np.asarray(values, dtype=np.float64)
            if self.config.features == "M"
            else self._expand_target(values)
        )
        if self.config.features == "M":
            return restored
        return restored[:, :, [self.input_columns.index(c) for c in self.target_columns]]

    def _expand_target(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("target values must have shape [batch, horizon, features]")
        expanded = np.zeros(
            (*values.shape[:2], len(self.input_columns)), dtype=np.float64
        )
        target_indices = [
            self.input_columns.index(column) for column in self.target_columns
        ]
        expanded[:, :, target_indices] = values
        return expanded


def build_data_bundle(config: DataConfig) -> DataBundle:
    """Read, clean, split, normalize, and window one forecasting dataset."""
    frame = _read_frame(config)
    input_columns, target_columns = _resolve_columns(frame, config)
    raw_values = frame[input_columns].to_numpy(dtype=np.float64)

    n_time = len(frame)
    train_end, val_end = _resolve_split_points(n_time, config)
    if train_end < config.seq_len + config.pred_len:
        raise ValueError("The training split is shorter than seq_len + pred_len")
    if val_end <= train_end or n_time <= val_end:
        raise ValueError("The dataset is too short for train/validation/test splits")

    scaler = Standardizer(config.scaler).fit(raw_values[:train_end])
    values = scaler.transform(raw_values).astype(np.float32)
    target_indices = np.asarray(
        [input_columns.index(column) for column in target_columns], dtype=np.int64
    )

    time_values = (
        frame[config.timestamp_col].to_numpy()
        if config.timestamp_col is not None
        else np.arange(n_time)
    )
    calendar = (
        _make_time_features(frame[config.timestamp_col])
        if config.add_time_features
        else None
    )

    split_ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_time),
    }
    datasets: dict[str, ForecastDataset] = {}
    for name, (split_start, split_end) in split_ranges.items():
        context_start = 0 if name == "train" else max(0, split_start - config.seq_len)
        latest_start = split_end - config.seq_len - config.pred_len
        starts = np.arange(
            context_start, latest_start + 1, config.stride, dtype=np.int64
        )
        starts = starts[
            starts + config.seq_len >= split_start
        ]
        if len(starts) == 0:
            raise ValueError(f"Split '{name}' contains no valid forecasting windows")
        datasets[name] = ForecastDataset(
            values=values,
            time_values=time_values,
            starts=starts,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            label_len=config.label_len,
            target_indices=target_indices,
            time_features=calendar,
        )

    return DataBundle(
        config=config,
        frame=frame,
        input_columns=input_columns,
        target_columns=target_columns,
        scaler=scaler,
        datasets=datasets,
        split_points={
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, n_time),
        },
    )


def _read_frame(config: DataConfig) -> pd.DataFrame:
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The input data file is empty")

    if config.timestamp_col is not None:
        if config.timestamp_col not in frame.columns:
            raise ValueError(f"Missing timestamp column: {config.timestamp_col}")
        frame[config.timestamp_col] = pd.to_datetime(
            frame[config.timestamp_col], errors="raise"
        )
        frame = (
            frame.sort_values(config.timestamp_col, kind="stable")
            .drop_duplicates(config.timestamp_col, keep="last")
            .reset_index(drop=True)
        )

    feature_columns = [
        column for column in frame.columns if column != config.timestamp_col
    ]
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(axis=1, how="all")

    numeric_columns = [
        column for column in frame.columns if column != config.timestamp_col
    ]
    if not numeric_columns:
        raise ValueError("No numeric time-series columns were found")
    missing = frame[numeric_columns].isna()
    if missing.any().any():
        if config.missing_policy == "raise":
            bad_columns = missing.columns[missing.any()].tolist()
            raise ValueError(f"Missing values found in columns: {bad_columns}")
        frame[numeric_columns] = frame[numeric_columns].ffill().bfill()
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Input contains non-finite numeric values")
    return frame


def _resolve_columns(
    frame: pd.DataFrame, config: DataConfig
) -> tuple[list[str], list[str]]:
    numeric_columns = [
        column for column in frame.columns if column != config.timestamp_col
    ]
    if config.features == "M":
        return numeric_columns, numeric_columns
    if config.target not in numeric_columns:
        raise ValueError(f"Target column not found: {config.target}")
    if config.features == "S":
        return [config.target], [config.target]  # type: ignore[list-item]
    return numeric_columns, [config.target]  # type: ignore[list-item]


def _resolve_split_points(n_time: int, config: DataConfig) -> tuple[int, int]:
    if config.split_points is not None:
        train_end, val_end = config.split_points
    else:
        train_end = int(n_time * config.train_ratio)
        val_end = train_end + int(n_time * config.val_ratio)
    if not 0 < train_end < val_end < n_time:
        raise ValueError(
            f"Invalid split points {(train_end, val_end)} for length {n_time}"
        )
    return train_end, val_end


def _make_time_features(values: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(values)
    periods = [
        (timestamps.dt.month.to_numpy() - 1, 12.0),
        (timestamps.dt.day.to_numpy() - 1, 31.0),
        (timestamps.dt.dayofweek.to_numpy(), 7.0),
        (timestamps.dt.hour.to_numpy(), 24.0),
        (timestamps.dt.minute.to_numpy(), 60.0),
    ]
    features = []
    for value, period in periods:
        phase = 2.0 * np.pi * value / period
        features.extend([np.sin(phase), np.cos(phase)])
    return np.stack(features, axis=1).astype(np.float32)


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)
