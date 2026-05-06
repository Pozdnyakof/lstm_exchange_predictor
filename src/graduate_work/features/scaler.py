"""Стандартизатор признаков с обязательным fit на train-выборке."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StandardScaler:
    """Pure-numpy реализация Z-нормализации.

    Сделано отдельно от sklearn, чтобы хранить параметры в простом dict
    и сериализовать их вместе с чекпоинтом модели.
    """

    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    columns_: tuple[str, ...] = field(default_factory=tuple)

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "StandardScaler":
        self.columns_ = tuple(columns)
        for col in columns:
            values = df[col].to_numpy(dtype=np.float64, copy=False)
            mu = float(np.nanmean(values))
            sigma = float(np.nanstd(values, ddof=0))
            self.mean_[col] = mu
            self.std_[col] = sigma if sigma > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_:
            msg = "Scaler was not fitted"
            raise RuntimeError(msg)
        out = df.copy()
        for col in self.columns_:
            mu = self.mean_[col]
            sigma = self.std_[col]
            out[col] = (out[col].astype(float) - mu) / sigma
        return out

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.fit(df, columns)
        return self.transform(df)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_,
            "std": self.std_,
            "columns": list(self.columns_),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "StandardScaler":
        scaler = cls()
        scaler.mean_ = dict(payload["mean"])
        scaler.std_ = dict(payload["std"])
        scaler.columns_ = tuple(payload["columns"])
        return scaler
