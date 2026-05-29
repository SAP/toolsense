"""LiteLLM-based chain wrapper for structured LLM calls with Pydantic output parsing.

Proxy configuration (via .env or environment variables):
    LITELLM_BASE_URL  — base URL of a LiteLLM proxy (e.g., http://localhost:4000)
    LITELLM_API_KEY   — API key for the proxy (optional)

If LITELLM_BASE_URL is not set, calls go directly to the provider using the
standard provider API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).

Usage:
    chain = LLMChain(model="claude-4.5-sonnet", prompt=prompt_template, output_parser=parser)
    result = chain.invoke({"var1": "...", "var2": "..."})
    parsed_object = result["responses"][0]

LiteLLM model strings: https://docs.litellm.ai/docs/providers
"""

import logging
import os
import time

import litellm
from openai import OpenAI

logger = logging.getLogger(__name__)

_PROXY_BASE_URL = os.environ.get("LITELLM_BASE_URL", "").strip() or None
_PROXY_API_KEY = os.environ.get("LITELLM_API_KEY", "").strip() or None


class LLMChain:
    """LiteLLM-based chain with a LangChain-compatible PromptTemplate and Pydantic output parser."""

    def __init__(self, model: str, prompt, output_parser, max_retries: int = 3, temperature: float = 0.7):
        """
        Args:
            model: LiteLLM model string (e.g., "openai/gpt-4o", "claude-4.5-sonnet").
                   When using a proxy, use the model alias configured in your proxy config.
            prompt: langchain_core.prompts.PromptTemplate instance.
            output_parser: langchain_core.output_parsers.PydanticOutputParser instance.
            max_retries: Retry count on parsing/API failure.
            temperature: LLM sampling temperature.
        """
        self.model = model
        self.prompt = prompt
        self.output_parser = output_parser
        self.max_retries = max_retries
        self.temperature = temperature

    def invoke(self, inputs: dict, config=None) -> dict:
        """Format the prompt, call LiteLLM, parse and return structured result.

        Returns:
            dict with key "responses" containing a list with the parsed Pydantic object.
        """
        formatted = self.prompt.format(**inputs)
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                if _PROXY_BASE_URL:
                    client = OpenAI(base_url=_PROXY_BASE_URL, api_key=_PROXY_API_KEY or "unused")
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": formatted}],
                        temperature=self.temperature,
                    )
                else:
                    response = litellm.completion(
                        model=self.model,
                        messages=[{"role": "user", "content": formatted}],
                        temperature=self.temperature,
                    )
                content = response.choices[0].message.content
                parsed = self.output_parser.parse(content)
                return {"responses": [parsed]}
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    logger.warning("LLM chain attempt %d failed: %s. Retrying...", attempt + 1, e)
                    time.sleep(2**attempt)
        raise last_exc
