import json

class DataHealthGenerator:
    """
    Responsible for:
    - Preparing the Data Health prompt
    - Calling LLMClient to generate the final report text
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

        self.system_prompt = """
        You are a senior data quality analyst.

        You will receive a JSON object that contains:
        - A list of data quality checks
        - Each check includes:
          - name: the check name
          - status: "healthy" | "warning" | "critical"
          - details: metrics and detected issues

        Your task:
        1. Analyze all checks and produce a clear, concise Data Quality Report.
        2. Keep the tone professional and factual.
        3. Highlight only the important issues (missing values, outliers, duplicates, inconsistent data types, empty dataset, rare categories…).
        4. Organize your answer in this format:

        ### Overall Quality Status:
        - A short (1–2 sentence) summary of the dataset's health.

        ### Key Issues:
        - Bullet points describing detected problems (from all checks).
        - Each point should explain:
          - what the issue is
          - where it occurs (columns)
          - why it matters

        ### Recommended Fixes:
        - For each issue category,
          give 1–2 practical suggestions for how to fix it.

        Output Format:
        - Clean markdown text only.
        - Do NOT generate JSON.
        - Do NOT invent issues not present in the input.
        """

    def generate(self, health_report: dict) -> str:
        """
        Takes the health report dict and returns
        LLM-generated data quality analysis.
        """

        user_prompt = (
            "Data Quality Checks JSON:\n"
            f"{json.dumps(health_report, separators=(',', ':'))}"
        )

        return self.llm_client.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0
        )
