"""
Easy tier — generation node.

Generates concise, enterprise-style queries that map to EXACTLY ONE tool.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator

from ...llm_utils import LLMChain


class EasySample(BaseModel):
    query: str = Field(..., description="A concise, enterprise-style user query pointing to exactly one tool.")
    answer: list[str] = Field(..., description="List containing exactly 1 tool name.")

    @model_validator(mode="after")
    def validate_answer(self) -> "EasySample":
        if len(self.answer) != 1:
            raise ValueError(f"Easy tier requires exactly 1 answer tool name, got {len(self.answer)}")
        return self


class EasyBatch(BaseModel):
    samples: list[EasySample] = Field(..., description="List of generated easy-tier eval samples.")

    @model_validator(mode="after")
    def check_count(self) -> "EasyBatch":
        if not self.samples:
            raise ValueError("Generated batch is empty.")
        return self


_PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "prompts" / "easy_generate.jinja2"
_parser = PydanticOutputParser(pydantic_object=EasyBatch)


class EasyGenerateNode:
    def __init__(self, model: str, n_samples: int = 3):
        self._n_samples = n_samples
        prompt = PromptTemplate(
            template=_PROMPT_TEMPLATE_PATH.read_text(),
            input_variables=[
                "anchor_tool_name", "anchor_tool_description",
                "candidate_pool_str", "n_samples", "format_instructions", "previous_feedback",
            ],
            template_format="jinja2",
            partial_variables={"format_instructions": _parser.get_format_instructions()},
        )
        self._chain = LLMChain(model=model, prompt=prompt, output_parser=_parser)

    def __call__(self, state: dict) -> dict:
        seed_tool: dict = state["seed_tool"]
        candidate_pool: list[dict] = state["candidate_pool"]
        previous_feedback: str = state.get("easy_generation_feedback", "")

        pool_lines = [f"- {t['tool_name']}: {t.get('tool_description', '')}" for t in candidate_pool]
        candidate_pool_str = "\n".join(pool_lines)

        inputs = {
            "anchor_tool_name": seed_tool["tool_name"],
            "anchor_tool_description": seed_tool.get("tool_description", ""),
            "candidate_pool_str": candidate_pool_str,
            "n_samples": self._n_samples,
            "previous_feedback": previous_feedback or "None",
        }

        try:
            chain_output = self._chain.invoke(inputs)
            batch: EasyBatch = chain_output["responses"][0]
        except Exception as e:
            return {"easy_generation_error": str(e), "easy_raw_samples": []}

        raw_samples = [
            {
                "sample_id": str(uuid.uuid4()),
                "query": s.query,
                "answer": s.answer,
                "complexity": "easy",
                "_validated": False,
            }
            for s in batch.samples
        ]
        return {"easy_raw_samples": raw_samples, "easy_generation_error": None}
