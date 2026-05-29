"""
Generate MCQ (Multiple-Choice Question) probing benchmark for tool token internalization.

For each tool, asks an LLM to write a specific factual question about the tool's properties
(domain, input type, output type, capability, etc.) and generate one correct short answer
plus three plausible-but-wrong alternatives. At inference time the model sees only the
virtual token + the question + four options — it must answer from token semantics alone.

This is harder than description-matching MCQ: the model must understand *what the tool does*
well enough to answer a specific factual question, not just recall a full description.

Random baseline: 25% (4-way uniform).

Usage:
    python generate_mcq.py \\
        --tools-file tools.jsonl \\
        --output mcq_benchmark/mcq_data.jsonl \\
        --model openai/gpt-4o

    # Limit to first N tools (smoke test):
    python generate_mcq.py \\
        --tools-file tools.jsonl \\
        --output mcq_benchmark/mcq_data.jsonl \\
        --model claude-4.5-sonnet \\
        --num-samples 20

Outputs:
    mcq_data.jsonl  — one record per tool: question + correct_answer + wrong_answers + tool dict
    data_card.md    — generation parameters and statistics

Tool file format (JSONL, one tool per line):
    {"tool_name": "...", "tool_description": "...", ...}

LiteLLM model strings: https://docs.litellm.ai/docs/providers
"""

import argparse
import json
import logging
import time
import uuid
from pathlib import Path

import litellm
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()

import os

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_PROXY_BASE_URL = os.environ.get("LITELLM_BASE_URL", "").strip() or None
_PROXY_API_KEY = os.environ.get("LITELLM_API_KEY", "").strip() or None

_MCQ_JUDGE_PROMPT = """\
You are validating a multiple-choice question entry for an AI tool probing benchmark.

Tool description:
{description}

Generated entry:
  Question      : {question}
  Correct answer: {correct_answer}
  Wrong answers : {wrong_answers}

Check ALL of the following:
1. The question is specific to THIS tool — not generically answerable for any API.
2. The correct answer is directly and unambiguously supported by the description.
3. Each of the three wrong answers is plausible for a tool in the same domain but clearly incorrect for THIS tool.
4. All four options are meaningfully distinct from each other.
5. The question uses "this tool" as a placeholder — the actual tool name does not appear.

Set accept=true only if ALL five checks pass. Otherwise set accept=false and state which check failed.

{format_instructions}"""

_MCQ_PROMPT = """\
You are building a multiple-choice probing benchmark for AI tool retrieval research.

Below is the description of an API or software tool:
{description}

Your task: generate one specific factual question about this tool's properties and provide
one correct short answer plus three wrong-but-plausible alternatives.

Rules:
1. Use "this tool" in the question — never include the actual tool name or service name.
   (The model will see only a virtual token at inference time, so the name must not be a hint.)
2. The question must be specific to THIS tool — not answerable for a generic API.
   Good topics: primary output type, domain/industry, key input, core capability, supported format.
   Bad question: "Does this tool provide an API?" (too generic).
3. Each answer option must be a short phrase (2–8 words), not a full sentence.
4. The correct_answer must be directly supported by the description above.
5. The three wrong answers must be plausible alternatives — the kind of answer a user might
   expect from a tool in the same domain — but clearly incorrect for THIS specific tool.
6. All four options (correct + wrong) must be meaningfully distinct from each other.
7. Set skip=true if you cannot form a specific, unambiguous factual question from this description.

{format_instructions}"""

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")


class JudgeVerdict(BaseModel):
    accept: bool = Field(..., description="True if the entry passes all quality checks")
    reason: str = Field(..., description="One sentence: which check passed/failed and why")


class MCQItem(BaseModel):
    question: str = Field(
        ..., description="Factual question about the tool using 'this tool' as placeholder"
    )
    correct_answer: str = Field(
        ..., description="Short phrase (2–8 words) that correctly answers the question"
    )
    wrong_answers: list[str] = Field(
        ..., description="Exactly 3 short phrases that are plausible but wrong for this tool"
    )
    skip: bool = Field(
        ..., description="True if no specific, unambiguous factual question can be formed"
    )
    reasoning: str = Field(
        ..., description="One sentence explaining the question or why skipped"
    )

    @field_validator("wrong_answers")
    @classmethod
    def validate_wrong_answers_count(cls, v: list[str]) -> list[str]:
        if not v:
            return v
        if len(v) != 3:
            raise ValueError(f"wrong_answers must have exactly 3 elements, got {len(v)}")
        return v


def _load_tools(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _call_llm(model: str, prompt_template, output_parser, inputs: dict, max_retries: int = 3,
              temperature: float = 0.7):
    formatted = prompt_template.format(**inputs)
    last_exc = None
    for attempt in range(max_retries):
        try:
            if _PROXY_BASE_URL:
                client = OpenAI(base_url=_PROXY_BASE_URL, api_key=_PROXY_API_KEY or "unused")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted}],
                    temperature=temperature,
                )
            else:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": formatted}],
                    temperature=temperature,
                )
            content = response.choices[0].message.content
            return output_parser.parse(content)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                logger.warning("Attempt %d failed: %s. Retrying...", attempt + 1, e)
                time.sleep(2**attempt)
    raise last_exc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tools-file", required=True, help="Input tools JSONL file")
    parser.add_argument("--output", required=True, help="Output mcq_data.jsonl path")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LiteLLM model string (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Model for LLM judge validation (default: same as --model)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Max number of tools to process (default: all)"
    )
    args = parser.parse_args()
    judge_model = args.judge_model or args.model

    records = _load_tools(args.tools_file)
    if args.num_samples is not None:
        records = records[: args.num_samples]
    print(f"Loaded {len(records)} tools from {args.tools_file}")
    print(f"Using model: {args.model}  Judge model: {judge_model}")

    mcq_parser = PydanticOutputParser(pydantic_object=MCQItem)
    prompt = PromptTemplate(
        template=_MCQ_PROMPT,
        input_variables=["description"],
        partial_variables={"format_instructions": mcq_parser.get_format_instructions()},
    )

    judge_parser = PydanticOutputParser(pydantic_object=JudgeVerdict)
    judge_prompt = PromptTemplate(
        template=_MCQ_JUDGE_PROMPT,
        input_variables=["description", "question", "correct_answer", "wrong_answers"],
        partial_variables={"format_instructions": judge_parser.get_format_instructions()},
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = rejected = errors = 0

    with open(out_path, "w") as f:
        for i, record in enumerate(records):
            description = record.get("tool_description", "")
            tool_name = record.get("tool_name", f"record_{i}")

            try:
                item: MCQItem = _call_llm(
                    args.model, prompt, mcq_parser,
                    {"description": description},
                )
            except Exception as e:
                print(f"  [{i+1}/{len(records)}] LLM error for {tool_name!r}: {e}")
                errors += 1
                continue

            if item.skip:
                skipped += 1
                continue

            try:
                verdict: JudgeVerdict = _call_llm(
                    judge_model, judge_prompt, judge_parser,
                    {
                        "description": description,
                        "question": item.question,
                        "correct_answer": item.correct_answer,
                        "wrong_answers": ", ".join(item.wrong_answers),
                    },
                    temperature=0.0,
                )
            except Exception as e:
                logger.warning("Judge error for %r: %s. Skipping item.", tool_name, e)
                errors += 1
                continue

            if not verdict.accept:
                rejected += 1
                continue

            out_record = {
                "id": str(uuid.uuid4()),
                "question": item.question,
                "correct_answer": item.correct_answer,
                "wrong_answers": item.wrong_answers,
                "tool": record,
            }
            f.write(json.dumps(out_record) + "\n")
            kept += 1

            if (i + 1) % 20 == 0 or i + 1 == len(records):
                print(
                    f"  [{i+1}/{len(records)}] kept={kept}  skipped={skipped}  "
                    f"rejected={rejected}  errors={errors}"
                )

    benchmark_version = out_path.parent.name
    data_card = f"""\
# MCQ Probing Benchmark {benchmark_version}

4-way multiple-choice: given a tool's virtual token, answer a specific factual question
about the tool's properties (output type, domain, input, capability, etc.).
The model must select the correct short-phrase answer (A/B/C/D) from token semantics alone.

Random baseline: 25%.

## Parameters

| Parameter | Value |
|---|---|
| source | `{args.tools_file}` |
| num_samples | `{args.num_samples if args.num_samples is not None else "all"}` |
| llm_model | `{args.model}` |
| judge_model | `{judge_model}` |

## Statistics

| Stat | Value |
|---|---|
| input_tools | {len(records)} |
| output_tools | {kept} |
| skipped_no_question | {skipped} |
| judge_rejected | {rejected} |
| llm_errors | {errors} |

## Data Fields (`mcq_data.jsonl`)

- `id` — UUID
- `question` — factual question about the tool using "this tool" as placeholder
- `correct_answer` — short phrase correctly answering the question
- `wrong_answers` — list of 3 plausible but incorrect short-phrase alternatives
- `tool` — full tool dict (used to derive the virtual token at eval time)
"""
    (out_path.parent / "data_card.md").write_text(data_card)

    print(f"\nWritten: {kept}  Skipped: {skipped}  Rejected: {rejected}  Errors: {errors}")
    print(f"MCQ benchmark  → {out_path}")
    print(f"Data card      → {out_path.parent / 'data_card.md'}")


if __name__ == "__main__":
    main()
