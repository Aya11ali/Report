import pandas as pd
import numpy as np

class SchemaExtractor:

    def extract_schema(self, df: pd.DataFrame) -> dict:
        schema = {}

        for col in df.columns:
            series = df[col]

            column_info = {
                "column_name": col,
                "dtype_inferred": self.infer_type(series),
                "nullable": bool(series.isnull().any()),
            }

            schema[col] = column_info

        return schema

    def infer_type(self, series: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "float"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "string"

    def get_samples(self, series: pd.Series, n=5):
        return series.dropna().sample(min(n, len(series)), random_state=1).tolist()

    def normalize(self, value):
        """Convert numpy types → Python native so JSON can serialize."""

        # Case 1 — list → normalize each element
        if isinstance(value, list):
            return [self.normalize(v) for v in value]

        # Case 2 — numpy array → convert to list then normalize
        if isinstance(value, np.ndarray):
            return self.normalize(value.tolist())

        # Case 3 — pandas or numpy scalar
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float64)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)

        # Case 4 — pandas Timestamp
        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        # Case 5 — NaN / None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass  # skip checking for lists

        # Case 6 — anything else
        return value
