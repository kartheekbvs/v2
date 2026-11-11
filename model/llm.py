
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLM:
    def __init__(self):
        base_url = os.getenv("base_url")
        if base_url:
            base_url = base_url.strip()

        if not base_url or base_url.lower() in {"none", "null", ""}:
            raise ValueError("base_url is missing from .env")

        def _is_placeholder(value):
            if value is None:
                return True
            value = value.strip().strip('"\'')
            return value.lower() in {
                "",
                "none",
                "null",
                "placeholder",
                "replace-with-your-provider-api-key",
            }

        api_key = None
        for candidate in (
            os.getenv("api_key"),
            os.getenv("polli_key"),
            os.getenv("OPENAI_API_KEY"),
        ):
            if candidate is not None and not _is_placeholder(candidate):
                api_key = candidate.strip().strip('"\'')
                break

        if not api_key:
            raise ValueError(
                "No valid API key found. Set api_key=... or polli_key=... in .env"
            )

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={"ngrok-skip-browser-warning": "true"},
            max_retries=3,
            timeout=60.0,
        )
        self.model = os.getenv("LLM_MODEL")

    def generate(self, messages, tools=None):
        request = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        return self.client.chat.completions.create(**request)
