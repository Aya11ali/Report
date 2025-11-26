from utils.data_health_base import BaseHealthCheck, HealthCheckResult
import pandas as pd

class EmptyDatasetCheck(BaseHealthCheck):
    """
    Checks whether the dataset is empty:
    - No rows
    - OR zero columns
    """

    def run(self, df: pd.DataFrame) -> HealthCheckResult:

        is_empty = df.empty
        row_count = len(df)
        column_count = len(df.columns)

        if is_empty:
            status = "critical"
            message = "Dataset is completely empty. No analysis can be performed."
        elif row_count == 0:
            status = "critical"
            message = "Dataset has columns but contains zero rows."
        else:
            status = "healthy"
            message = "Dataset has valid rows and columns."

        return HealthCheckResult(
            name="Empty Dataset Check",
            status=status,
            details={
                "row_count": row_count,
                "column_count": column_count,
                "is_empty": is_empty,
                "message": message
            }
        )

class NullRatioCheck(BaseHealthCheck):
    """Calculates the null percentage for each column."""

    def __init__(self, warning_threshold=0.2, critical_threshold=0.5):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def run(self, df) -> HealthCheckResult:
        null_ratios = df.isna().mean().to_dict()
        max_null = max(null_ratios.values()) if null_ratios else 0

        if max_null >= self.critical_threshold:
            status = "critical"
        elif max_null >= self.warning_threshold:
            status = "warning"
        else:
            status = "healthy"

        return HealthCheckResult(
            name="Null Ratio Check",
            status=status,
            details={
                "null_ratio_per_column": null_ratios,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold
            }
        )

class DuplicateRowsCheck(BaseHealthCheck):
    """
    Detects duplicate rows in the dataset.
    Returns:
        - duplicate_count
        - duplicate_percentage
        - sample_duplicates (first 5 duplicate rows)
    """

    def __init__(self, warning_threshold=0.05, critical_threshold=0.2):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def run(self, df: pd.DataFrame) -> HealthCheckResult:

        duplicate_mask = df.duplicated()
        duplicate_count = int(duplicate_mask.sum())
        duplicate_percentage = float(duplicate_count / len(df)) if len(df) > 0 else 0

        # status decision
        if duplicate_percentage >= self.critical_threshold:
            status = "critical"
        elif duplicate_percentage >= self.warning_threshold:
            status = "warning"
        else:
            status = "healthy"

        # get small sample of duplicate rows
        sample_duplicates = df[duplicate_mask].head().to_dict(orient="records")

        return HealthCheckResult(
            name="Duplicate Rows Check",
            status=status,
            details={
                "duplicate_count": duplicate_count,
                "duplicate_percentage": round(duplicate_percentage, 4),
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "sample_duplicates": sample_duplicates
            }
        )

class OutlierIQRCheck(BaseHealthCheck):
    """
    Detects outliers in numeric columns using the IQR method.
    """

    def __init__(self, warning_threshold=0.05, critical_threshold=0.1, sample_size=5):

        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.sample_size = sample_size

    def run(self, df: pd.DataFrame) -> HealthCheckResult:
        numeric_cols = df.select_dtypes(include='number').columns
        numeric_cols = [c for c in numeric_cols if df[c].nunique() > 2]  # فقط الأعمدة المتنوعة
        outlier_counts = {}
        outlier_samples = {}

        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            mask = (df[col] < lower) | (df[col] > upper)
            count = int(mask.sum())
            outlier_counts[col] = count
            outlier_samples[col] = df.loc[mask].head(self.sample_size).to_dict(orient='records')

        max_outlier_pct = max((count/len(df) for count in outlier_counts.values()), default=0)

        if max_outlier_pct >= self.critical_threshold:
            status = "critical"
        elif max_outlier_pct >= self.warning_threshold:
            status = "warning"
        else:
            status = "healthy"

        return HealthCheckResult(
            name="Outlier IQR Check",
            status=status,
            details={
                "outlier_count_per_column": outlier_counts,
                # "outlier_samples_per_column": outlier_samples,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold
            }
        )
