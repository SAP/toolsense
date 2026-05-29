"""
Easy tier — validation node.

Two-stage validation:
1. Programmatic: all answer tool names must exist in the candidate pool.
2. LLM judge: query quality (concise, business language, correct ambiguity level).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ...llm_utils import LLMChain

_VALIDATOR_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "validator.jinja2"
_MAX_RETRIES_BEFORE_GIVE_UP = 3


class ValidationVerdict(BaseModel):
    validation_result: bool = Field(..., description="True if the sample passes all quality checks.")
    validation_reason: str = Field(..., description="Brief explanation of failure. Empty string if passed.")


class BatchValidationVerdict(BaseModel):
    verdicts: list[ValidationVerdict] = Field(
        ..., description="One verdict per sample, in the same order as input."
    )


def _programmatic_check(sample: dict, candidate_pool: list[dict]) -> tuple[bool, str]:
    pool_names = {t["tool_name"] for t in candidate_pool}
    for name in sample.get("answer", []):
        if name not in pool_names:
            return False, f"Hallucinated tool: '{name}' not in candidate pool."
    return True, "ok"


class EasyValidateNode:
    def __init__(self, model: str):
        parser = PydanticOutputParser(pydantic_object=BatchValidationVerdict)
        prompt = PromptTemplate(
            template=_VALIDATOR_PROMPT_PATH.read_text(),
            input_variables=["samples_str", "format_instructions", "complexity"],
            template_format="jinja2",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self._llm_judge = LLMChain(model=model, prompt=prompt, output_parser=parser)

    def __call__(self, state: dict) -> dict:
        raw_samples: list[dict] = state.get("easy_raw_samples", [])
        candidate_pool: list[dict] = state.get("candidate_pool", [])
        retry_count: int = state.get("easy_retry_count", 0)

        if not raw_samples:
            if retry_count >= _MAX_RETRIES_BEFORE_GIVE_UP:
                return {"easy_done": True}
            return {"easy_retry_count": retry_count + 1}

        prog_passed, prog_failed = [], []
        for s in raw_samples:
            ok, reason = _programmatic_check(s, candidate_pool)
            (prog_passed if ok else prog_failed).append(s if ok else (s, reason))

        llm_passed, llm_failed = [], []
        if prog_passed:
            samples_str = "\n\n".join(
                f"[Sample {i+1}]\nQuery: {s['query']}\nAnswer: {s['answer']}"
                for i, s in enumerate(prog_passed)
            )
            try:
                chain_output = self._llm_judge.invoke(
                    {"samples_str": samples_str, "complexity": "easy"}
                )
                verdict: BatchValidationVerdict = chain_output["responses"][0]
                for sample, v in zip(prog_passed, verdict.verdicts):
                    if v.validation_result:
                        llm_passed.append(sample)
                    else:
                        llm_failed.append((sample, v.validation_reason))
            except Exception:
                llm_passed = prog_passed

        if llm_passed:
            tool_map = {t["tool_name"]: t for t in candidate_pool}
            enriched = []
            for s in llm_passed:
                s["tools"] = [tool_map[n] for n in s["answer"] if n in tool_map]
                s["candidate_pool_ids"] = [t["tool_name"] for t in candidate_pool]
                s.pop("_validated", None)
                enriched.append(s)
            return {"completed_samples": enriched, "easy_done": True, "easy_samples_count": len(enriched)}

        all_reasons = [r for _, r in prog_failed] + [r for _, r in llm_failed]
        feedback = "Previous batch issues:\n" + "\n".join(f"- {r}" for r in all_reasons[:5])
        if retry_count >= _MAX_RETRIES_BEFORE_GIVE_UP:
            return {"easy_done": True}
        return {"easy_retry_count": retry_count + 1, "easy_generation_feedback": feedback}
