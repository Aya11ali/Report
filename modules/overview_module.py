from utils.info_extraction.metadata_extractor import DatasetMetadataExtractor
from utils.info_extraction.schema_extractor import SchemaExtractor
from utils.info_extraction.statistics_extractor import StatisticsExtractor
from utils.info_extraction.dataset_profile_builder import CombinedDatasetProfileBuilder
from utils.overview_generator import OverviewGenerator
from utils.llm_client import LLMClient

class OverviewModule:
    """
    Responsible for generating the final Overview section.
    Handles:
    1. Building dataset profile
    2. Connecting with LLM via OverviewGenerator
    3. Returning the final overview text
    """

    def __init__(self, llm_client: LLMClient):
        self.metadata_extractor = DatasetMetadataExtractor()
        self.schema_extractor = SchemaExtractor()
        self.stats_extractor = StatisticsExtractor()
        self.profile_builder = CombinedDatasetProfileBuilder(
            metadata_extractor=self.metadata_extractor,
            schema_extractor=self.schema_extractor,
            stats_extractor=self.stats_extractor
        )
        self.overview_generator = OverviewGenerator(llm_client=llm_client)

    def get_overview(self, df, file_path: str = None, n_samples: int = 5) -> str:
        """
        Main method to get the Overview section.
        Steps:
        1. Build the dataset profile
        2. Pass it to OverviewGenerator
        3. Return generated overview text
        """
        profile = self.profile_builder.build_profile(df, file_path=file_path, n_samples=n_samples)

        overview_text = self.overview_generator.generate(profile)

        return overview_text
