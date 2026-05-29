"""
Generate yes/no Q&A probing benchmark for tool token internalization.

For each tool in the input file, asks an LLM to generate one specific yes/no question
about the tool's properties (domain, input type, capability, output format, etc.). The
question uses "this tool" as a placeholder — never the actual tool name — so at inference
time the model must answer from token semantics alone, not from a name hint.

Tools where no meaningful question can be formed (too vague/generic) are discarded.

Usage:
    python generate_qa.py \\
        --tools-file tools.jsonl \\
        --output qa_benchmark/qa_data.jsonl \\
        --model openai/gpt-4o

    # Limit to first N tools (smoke test):
    python generate_qa.py \\
        --tools-file tools.jsonl \\
        --output qa_benchmark/qa_data.jsonl \\
        --model claude-4.5-sonnet \\
        --num-samples 20

Outputs:
    qa_data.jsonl  — one record per kept tool: question + answer + tool dict
    data_card.md   — generation parameters and statistics

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
from typing import Literal

import litellm
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

import os

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Proxy config — read by litellm automatically when set in the environment.
# Set LITELLM_BASE_URL and LITELLM_API_KEY in .env to route through a proxy.
_PROXY_BASE_URL = os.environ.get("LITELLM_BASE_URL", "").strip() or None
_PROXY_API_KEY = os.environ.get("LITELLM_API_KEY", "").strip() or None

_QA_JUDGE_PROMPT = """\
You are validating a yes/no Q&A entry for an AI tool probing benchmark.

Tool description:
{description}

Generated entry:
  Question : {question}
  Answer   : {answer}

Check ALL of the following:
1. The question is specific to THIS tool — not generically answerable for any API.
2. The answer ("{answer}") is directly and unambiguously supported by the description.
3. The question uses "this tool" as a placeholder — the actual tool name does not appear.
4. The question tests a verifiable property (domain, capability, input/output type, format, etc.).

Set accept=true only if ALL four checks pass. Otherwise set accept=false and state which check failed.

{format_instructions}"""

_QA_PROMPT = """\
You are building a factual probing benchmark for AI tool retrieval research.

Below is the description of an API or software tool:
{description}

Your task: generate one yes/no question that tests knowledge of a specific, verifiable \
functionality or capability of this tool. The correct answer MUST be "{target_answer}".

Rules:
1. The answer to your question must be exactly "{target_answer}" based on the description.
2. Use "this tool" in the question — never include the actual tool name or service name.
   (The model will see only a token at inference time, so the name must not be a hint.)
3. The question must be specific to THIS tool. "Does this tool provide an API?" is too generic.
   Good examples for Yes: "Does this tool process image inputs?", "Is this tool designed for financial data?"
   Good examples for No:  "Does this tool support voice/audio input?", "Does this tool return results in XML?"
   For No questions: ask about a plausible capability or data type that the description does NOT mention
   (e.g. a different modality, format, or domain that a user might expect but this tool doesn't handle).
4. Set skip=true if you cannot form a specific, unambiguous question with the required answer.

{format_instructions}"""

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")


class JudgeVerdict(BaseModel):
    accept: bool = Field(..., description="True if the entry passes all quality checks")
    reason: str = Field(..., description="One sentence: which check passed/failed and why")


class QAPair(BaseModel):
    question: str = Field(..., description="Yes/no question about the tool, using 'this tool' as placeholder")
    answer: Literal["Yes", "No"] = Field(..., description="Correct answer: Yes or No")
    skip: bool = Field(..., description="True if no meaningful specific question can be formed")
    reasoning: str = Field(..., description="One sentence explaining the question or why skipped")


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
    parser.add_argument("--output", required=True, help="Output qa_data.jsonl path")
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

    qa_parser = PydanticOutputParser(pydantic_object=QAPair)
    prompt = PromptTemplate(
        template=_QA_PROMPT,
        input_variables=["description", "target_answer"],
        partial_variables={"format_instructions": qa_parser.get_format_instructions()},
    )

    judge_parser = PydanticOutputParser(pydantic_object=JudgeVerdict)
    judge_prompt = PromptTemplate(
        template=_QA_JUDGE_PROMPT,
        input_variables=["description", "question", "answer"],
        partial_variables={"format_instructions": judge_parser.get_format_instructions()},
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = rejected = errors = yes_count = no_count = 0
    _targets = ["Yes", "No"]

    with open(out_path, "w") as f:
        for i, record in enumerate(records):
            description = record.get("tool_description", "")
            tool_name = record.get("tool_name", f"record_{i}")
            target_answer = _targets[i % 2]

            try:
                qa: QAPair = _call_llm(
                    args.model, prompt, qa_parser,
                    {"description": description, "target_answer": target_answer},
                )
            except Exception as e:
                print(f"  [{i+1}/{len(records)}] LLM error for {tool_name!r}: {e}")
                errors += 1
                continue

            if qa.skip:
                skipped += 1
                continue

            try:
                verdict: JudgeVerdict = _call_llm(
                    judge_model, judge_prompt, judge_parser,
                    {"description": description, "question": qa.question, "answer": qa.answer},
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
                "question": qa.question,
                "answer": qa.answer,
                "tool": record,
            }
            f.write(json.dumps(out_record) + "\n")
            kept += 1
            if qa.answer == "Yes":
                yes_count += 1
            else:
                no_count += 1

            if (i + 1) % 20 == 0 or i + 1 == len(records):
                print(
                    f"  [{i+1}/{len(records)}] kept={kept}  skipped={skipped}  "
                    f"rejected={rejected}  errors={errors}  yes={yes_count}  no={no_count}"
                )

    benchmark_version = out_path.parent.name
    data_card = f"""\
# Q&A Probing Benchmark {benchmark_version}

Yes/no questions about specific tool properties. Questions use "this tool" as a placeholder —
the model must answer from token semantics alone, without a name hint in the question.

Random baseline: 50%.

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
| answer_yes | {yes_count} |
| answer_no | {no_count} |

## Data Fields (`qa_data.jsonl`)

- `id` — UUID
- `question` — yes/no question about the tool (uses "this tool", not the actual name)
- `answer` — correct answer: "Yes" or "No"
- `tool` — full tool dict (used to derive the virtual token at eval time)
"""
    (out_path.parent / "data_card.md").write_text(data_card)

    print(f"\nWritten: {kept}  Skipped: {skipped}  Rejected: {rejected}  Errors: {errors}")
    if kept:
        print(f"Answer distribution: Yes={yes_count} ({yes_count/kept:.0%})  No={no_count} ({no_count/kept:.0%})")
    print(f"Q&A benchmark  → {out_path}")
    print(f"Data card      → {out_path.parent / 'data_card.md'}")


if __name__ == "__main__":
    main()
