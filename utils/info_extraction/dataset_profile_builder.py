import pandas as pd

class CombinedDatasetProfileBuilder:
    def __init__(self, metadata_extractor, schema_extractor, stats_extractor):
        self.metadata_extractor = metadata_extractor
        self.schema_extractor = schema_extractor
        self.stats_extractor = stats_extractor

    def build_profile(self, df: pd.DataFrame, file_path: str = None, n_samples: int = 5) -> dict:
        profile = {}

        # 1️⃣ Metadata
        file_meta = self.metadata_extractor.extract_file_metadata(file_path) if file_path else {}
        df_meta = self.metadata_extractor.extract_dataframe_metadata(df)
        profile["metadata"] = {**file_meta, **df_meta}

        # 2️⃣ Schema
        profile["schema"] = self.schema_extractor.extract_schema(df)

        # 3️⃣ Sample rows
        profile["sample_rows"] = df.head(n_samples).to_dict(orient="records")

        # 4️⃣ Statistics
        profile["statistics"] = self.stats_extractor.extract(df)

        return profile
