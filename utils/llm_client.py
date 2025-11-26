import os
from openai import OpenAI
from config.config import API_KEY

class LLMClient:
    """
    A lightweight reusable client responsible ONLY for sending prompts 
    to the LLM and returning responses.
    """

    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY
        )

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0):
        """
        Sends a chat-completion request to the model.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )

        try:
            return response.choices[0].message.content.strip()
        except Exception:
            return "LLM response could not be parsed."
