import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()

API_KEY = config.get("OPENAI_API_KEY")
MODEL_NAME = config.get("MODEL_NAME")
