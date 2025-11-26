from typing import Dict 
import os
import pandas as pd

class DatasetMetadataExtractor:
    """
    SRP: Extract metadata for LLM consumption.
    Does NOT load the data.
    Does NOT infer schema.
    Only extracts metadata.
    """

    def extract_file_metadata(self, file_path: str) -> Dict:
        """Extracts metadata related to the file itself."""
        try:
            size_bytes = os.path.getsize(file_path)
        except OSError:
            size_bytes = None

        file_type = os.path.splitext(file_path)[1].lstrip('.').lower()

        return {
            "file_path": file_path,
            "file_type": file_type,
            "file_size_bytes": size_bytes
        }

    def extract_dataframe_metadata(self, df: pd.DataFrame) -> Dict:
        """Extracts metadata related to the DataFrame."""
        return {
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "missing_counts": {
                col: int(df[col].isna().sum()) for col in df.columns
            }
        }

    def extract_full_metadata(self, file_path: str, df: pd.DataFrame) -> Dict:
        """
        Combines file-level + dataframe-level metadata into single package
        ready to be sent to the LLM.
        """
        file_meta = self.extract_file_metadata(file_path)
        df_meta = self.extract_dataframe_metadata(df)

        return {
            **file_meta,
            **df_meta
        }
