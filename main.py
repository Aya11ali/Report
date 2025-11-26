import os
import sys
import pandas as pd
import json

project_path = "/content/drive/MyDrive/Colab Notebooks/report"
sys.path.append(project_path)

from utils.data_loader import CSVLoader
from utils.llm_client import LLMClient
from config.config import MODEL_NAME
from modules.overview_module import OverviewModule
from modules.data_health_module import DataHealthModule



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def main():
  data_path = "/content/drive/MyDrive/Colab Notebooks/report/data/Housing.csv"


  loader = CSVLoader(data_path)
  df = loader.load()

  llm_client = LLMClient(model=MODEL_NAME)

  # overview_module = OverviewModule(llm_client=llm_client)
  # overview_text = overview_module.get_overview(df, file_path=data_path)
  # print(overview_text)

  data_health_module = DataHealthModule(llm_client=llm_client)
  data_health_text = data_health_module.get_data_health(df)
  print(data_health_text)

if __name__ == "__main__":
  main()
