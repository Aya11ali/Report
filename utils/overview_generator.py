import json

class OverviewGenerator:
    """
    Responsible for:
    - Preparing the Overview prompt
    - Calling LLMClient to get the Overview text
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

        self.system_prompt = """
        You are a data analyst assistant.
        You are given a dataset profile in JSON format, which includes:
        - metadata: general info (rows, columns, missing values)
        - schema: column-level info (name, type, unique values, sample values)
        - sample_rows: few example rows
        - statistics: numeric/categorical summaries

        Your task:
        1. Generate a short, clear overview (3-5 sentences) describing the dataset.
        2. Focus on the meaning and context: what the dataset represents, what the entities/records are.
        3. Include only the most important insights.
        4. Mention missing values or imbalanced target if relevant.
        5. Format: a short paragraph + a bullet list of key insights.
        6. Return ONLY the formatted text.
        """

    def generate(self, dataset_profile: dict) -> str:
        """
        Takes the full dataset profile dict and returns
        the LLM-generated overview text.
        """

        user_prompt = f"Dataset profile:\n{json.dumps(dataset_profile, separators=(',', ':'))}"

        return self.llm_client.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0
        )
