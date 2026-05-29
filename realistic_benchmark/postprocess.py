"""postprocess.py — Convert raw generation output into clean eval records.

Reads samples.jsonl produced by run_generation.py and writes eval-ready JSONL
where each line matches the ToolRetrievalDataset format:
  - query:          the generated natural-language query
  - tool:           single tool dict (easy) or list of tool dicts (medium/hard)
  - analyzed_tools: candidate pool as list of full tool dicts (hard negatives + GT)
  - complexity:     "easy" | "medium" | "hard"
  - sample_id:      UUID for traceability

Usage:
    python -m realistic_benchmark.postprocess \\
        --input  output/samples.jsonl \\
        --output output/eval.jsonl \\
        --tools-file tools.jsonl

    # Filter by complexity tier:
    python -m realistic_benchmark.postprocess \\
        --input  output/samples.jsonl \\
        --output output/eval_easy.jsonl \\
        --tools-file tools.jsonl \\
        --complexity easy
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TOOL_STRIP_KEYS = {"pool_tier", "ibn_target"}


def _clean_tool(tool: dict) -> dict:
    return {k: v for k, v in tool.items() if k not in _TOOL_STRIP_KEYS}


def _load_tool_catalog(tools_file: str) -> dict[str, dict]:
    catalog = {}
    with open(tools_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tool = json.loads(line)
            catalog[tool["tool_name"]] = tool
    return catalog


def postprocess(
    input_path: Path,
    output_path: Path,
    tools_file: str,
    complexity: str | None = None,
) -> dict:
    """Convert raw generation output to eval-ready JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tool_catalog = _load_tool_catalog(tools_file)
    logger.info("Loaded tool catalog: %d tools from %s", len(tool_catalog), tools_file)

    total_read = written = skipped_filter = skipped_missing = missing_pool_names = 0
    seen_ids: set[str] = set()

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total_read += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped_missing += 1
                continue

            sid = sample.get("sample_id")
            if sid in seen_ids:
                skipped_missing += 1
                continue
            if sid:
                seen_ids.add(sid)

            if complexity and sample.get("complexity") != complexity:
                skipped_filter += 1
                continue

            if not all(k in sample for k in ("query", "answer", "tools", "candidate_pool_ids", "complexity")):
                skipped_missing += 1
                continue

            analyzed_tools = []
            for name in sample["candidate_pool_ids"]:
                full_tool = tool_catalog.get(name)
                if full_tool is None:
                    missing_pool_names += 1
                else:
                    analyzed_tools.append(_clean_tool(full_tool))

            answer_tools = [_clean_tool(t) for t in sample.get("tools", [])]
            tool_field = answer_tools[0] if len(answer_tools) == 1 else answer_tools

            record = {
                "sample_id": sid,
                "query": sample["query"],
                "tool": tool_field,
                "analyzed_tools": analyzed_tools,
                "complexity": sample["complexity"],
            }
            f_out.write(json.dumps(record) + "\n")
            written += 1

    return {
        "total_read": total_read,
        "written": written,
        "skipped_filter": skipped_filter,
        "skipped_missing_or_dup": skipped_missing,
        "missing_pool_tool_lookups": missing_pool_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-process generation output into eval-ready JSONL")
    parser.add_argument("--input", "-i", required=True, help="Path to samples.jsonl")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--tools-file", required=True, help="Tool catalog JSONL (for resolving candidate pool)")
    parser.add_argument(
        "--complexity", choices=["easy", "medium", "hard"], default=None,
        help="Only keep samples of this complexity tier (default: keep all)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Input:      {input_path}")
    print(f"Output:     {args.output}")
    print(f"Tools:      {args.tools_file}")
    if args.complexity:
        print(f"Filter:     complexity={args.complexity}")

    stats = postprocess(
        input_path=input_path,
        output_path=Path(args.output),
        tools_file=args.tools_file,
        complexity=args.complexity,
    )

    print(f"\nDone.")
    print(f"  Read:    {stats['total_read']}")
    print(f"  Written: {stats['written']}")
    if stats["skipped_filter"]:
        print(f"  Skipped (tier filter):  {stats['skipped_filter']}")
    if stats["skipped_missing_or_dup"]:
        print(f"  Skipped (missing/dup):  {stats['skipped_missing_or_dup']}")
    if stats["missing_pool_tool_lookups"]:
        print(f"  Warn: pool tool names not found in catalog: {stats['missing_pool_tool_lookups']}")


if __name__ == "__main__":
    main()
