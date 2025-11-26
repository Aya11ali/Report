from typing import Dict
import pandas as pd

class StatisticsExtractor:
    def extract(self, df: pd.DataFrame) -> dict:
        stats = {}
        for col in df.columns:
            series = df[col]
            col_stats = {}

            # Numeric
            if pd.api.types.is_numeric_dtype(series):
                col_stats = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "missing": int(series.isna().sum())
                }

            # Categorical / string
            elif pd.api.types.is_string_dtype(series):
                top = series.value_counts().head(3).to_dict()
                col_stats = {
                    "unique_values": int(series.nunique()),
                    "top_values": top,
                    "missing": int(series.isna().sum())
                }

            # Boolean
            elif pd.api.types.is_bool_dtype(series):
                counts = series.value_counts().to_dict()
                col_stats = {
                    "counts": {str(k): int(v) for k, v in counts.items()},
                    "missing": int(series.isna().sum())
                }

            stats[col] = col_stats
        return stats
