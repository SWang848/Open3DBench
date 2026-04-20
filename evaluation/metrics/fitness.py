from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _to_python_scalar(value):
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and pd.isna(value):
        return None

    if pd.isna(value):
        return None

    return value


def cal_fitness_score(df: pd.DataFrame, metrics: Sequence[str]):
    normalized_components = {}
    best_values = {}

    for metric in metrics:
        numeric_col = pd.to_numeric(df[metric], errors="coerce")
        col_min = numeric_col.min()
        col_max = numeric_col.max()

        if col_max <= 0:
            denom = abs(col_min)
            if denom != 0:
                normalized_series = numeric_col.abs() / denom
            else:
                normalized_series = numeric_col.where(numeric_col.isna(), 0.0)
            best_value = col_max
        elif col_min >= 0:
            denom = col_max
            if denom != 0:
                normalized_series = numeric_col / denom
            else:
                normalized_series = numeric_col.where(numeric_col.isna(), 0.0)
            best_value = col_min
        else:
            denom = max(abs(col_min), abs(col_max))
            if denom != 0:
                normalized_series = numeric_col.abs() / denom
            else:
                normalized_series = numeric_col.where(numeric_col.isna(), 0.0)
            abs_series = numeric_col.abs()
            best_index = abs_series.idxmin()
            best_value = numeric_col.loc[best_index]

        normalized_components[metric] = normalized_series.clip(lower=0.0, upper=1.0)
        best_values[metric] = _to_python_scalar(best_value)

    normalized_matrix = pd.DataFrame(normalized_components, index=df.index)
    result_df = df.copy()
    result_df["Fitness"] = np.sqrt((normalized_matrix ** 2).sum(axis=1, skipna=False))
    result_df = result_df.sort_values(by="Fitness", ascending=True, na_position="last").reset_index(drop=True)

    return result_df, best_values
