from utils.checks import NullRatioCheck, OutlierIQRCheck, EmptyDatasetCheck, DuplicateRowsCheck
from utils.data_health_base import HealthValidator

from utils.data_health_generator import DataHealthGenerator
from utils.llm_client import LLMClient

class DataHealthModule:
    """
    Responsible for generating the final Overview section.
    Handles:
    1. Building dataset profile
    2. Connecting with LLM via OverviewGenerator
    3. Returning the final overview text
    """

    def __init__(self, llm_client: LLMClient):
        self.null_ratio_check = NullRatioCheck()
        self.outlier_check = OutlierIQRCheck()
        self.empty_dataset_check = EmptyDatasetCheck()
        self.duplicate_rows_check = DuplicateRowsCheck()
        self.health_validator = HealthValidator([
            self.null_ratio_check,
            self.outlier_check,
            self.empty_dataset_check,
            self.duplicate_rows_check
        ])
        self.data_health_generator = DataHealthGenerator(llm_client=llm_client)
    def get_data_health(self, df) -> str:
      """
      Main method to generate the Data Health section.

      Steps:
      1. Run all health checks using HealthValidator
      2. Convert the report to dict
      3. Pass it to DataHealthGenerator
      4. Return the final generated text
      """

      health_report = self.health_validator.run(df)

      health_dict = health_report.to_dict()

      health_text = self.data_health_generator.generate(health_dict)

      return health_text
