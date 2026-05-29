"""
Medium tier — generation node.

Generates concise, enterprise-style queries genuinely ambiguous across 2-3 tools.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator

from ...llm_utils import LLMChain


class MediumSample(BaseModel):
    query: str = Field(..., description="A concise enterprise query genuinely ambiguous across 2-3 tools.")
    answer: list[str] = Field(..., description="List of 2 or 3 tool names that plausibly answer the query.")

    @model_validator(mode="after")
    def validate_answer(self) -> "MediumSample":
        if not (2 <= len(self.answer) <= 3):
            raise ValueError(f"Medium tier requires 2-3 answer tool names, got {len(self.answer)}")
        return self


class MediumBatch(BaseModel):
    samples: list[MediumSample] = Field(..., description="List of generated medium-tier eval samples.")

    @model_validator(mode="after")
    def check_count(self) -> "MediumBatch":
        if not self.samples:
            raise ValueError("Generated batch is empty.")
        return self


_PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "prompts" / "medium_generate.jinja2"
_parser = PydanticOutputParser(pydantic_object=MediumBatch)


class MediumGenerateNode:
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
        previous_feedback: str = state.get("medium_generation_feedback", "")

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
            batch: MediumBatch = chain_output["responses"][0]
        except Exception as e:
            return {"medium_generation_error": str(e), "medium_raw_samples": []}

        raw_samples = [
            {
                "sample_id": str(uuid.uuid4()),
                "query": s.query,
                "answer": s.answer,
                "complexity": "medium",
                "_validated": False,
            }
            for s in batch.samples
        ]
        return {"medium_raw_samples": raw_samples, "medium_generation_error": None}
