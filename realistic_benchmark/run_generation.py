"""run_generation.py — Entry point for the realistic benchmark generation pipeline.

Loads tools, builds the pipeline, and runs per-seed generation with async concurrency.

Usage:
    # Step 1: Prepare stratified seeds (run once)
    python -m realistic_benchmark.seed_preparation \\
        --tools-file tools.jsonl \\
        --n-seeds 1000 \\
        --output seeds/seeds.jsonl

    # Step 2: Single-seed smoke test
    python -m realistic_benchmark.run_generation test \\
        --tools-file tools.jsonl \\
        --seeds-file seeds/seeds.jsonl \\
        --model openai/gpt-4o

    # Step 3: Full batch generation
    python -m realistic_benchmark.run_generation generate \\
        --tools-file tools.jsonl \\
        --seeds-file seeds/seeds.jsonl \\
        --output-dir output/ \\
        --model openai/gpt-4o \\
        --concurrency 8

    # Step 4: Post-process into eval-ready JSONL
    python -m realistic_benchmark.run_generation postprocess \\
        --input output/samples.jsonl \\
        --output output/eval.jsonl \\
        --tools-file tools.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os

from .environment import make_initial_state
from .pipeline import build_pipeline

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")
SEED_TIMEOUT_SECONDS = 600


def _load_tools(tools_file: str) -> list[dict]:
    with open(tools_file) as f:
        tools = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(tools)} tools from {tools_file}")
    return tools


async def _process_seed(
    pipeline,
    seed: dict,
    semaphore: asyncio.Semaphore,
    timeout: int = SEED_TIMEOUT_SECONDS,
) -> tuple[dict, list[dict], str]:
    async with semaphore:
        try:
            initial_state = make_initial_state(seed)
            result = await asyncio.wait_for(
                asyncio.to_thread(pipeline.invoke, initial_state),
                timeout=timeout,
            )
            return seed, list(result.get("completed_samples", [])), "ok"
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] seed={seed.get('tool_name', '?')} exceeded {timeout}s")
            return seed, [], f"failed: timeout after {timeout}s"
        except Exception as e:
            import traceback
            print(f"[ERROR] seed={seed.get('tool_name', '?')}: {e}")
            traceback.print_exc()
            return seed, [], f"failed: {e}"


async def _run_generation_async(
    pipeline,
    seeds: list[dict],
    output_path: Path,
    concurrency: int,
    timeout: int = SEED_TIMEOUT_SECONDS,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    samples_path = output_path / "samples.jsonl"
    failed_path = output_path / "failed_seeds.jsonl"

    seeds_done = total_samples = failed_count = 0
    total = len(seeds)
    start_time = time.time()

    with open(samples_path, "w") as f_samples, open(failed_path, "w") as f_failed:
        tasks = [_process_seed(pipeline, seed, semaphore, timeout=timeout) for seed in seeds]
        for coro in asyncio.as_completed(tasks):
            seed, samples, status = await coro
            seeds_done += 1

            if status == "ok":
                for s in samples:
                    f_samples.write(json.dumps(s) + "\n")
                f_samples.flush()
                total_samples += len(samples)
            else:
                f_failed.write(json.dumps({"seed": seed, "error": status}) + "\n")
                f_failed.flush()
                failed_count += 1

            elapsed = time.time() - start_time
            rate = seeds_done / elapsed if elapsed > 0 else 0
            eta = (total - seeds_done) / rate if rate > 0 else 0

            if seeds_done % 10 == 0 or seeds_done == total:
                avg = total_samples / max(seeds_done - failed_count, 1)
                print(
                    f"  [{seeds_done}/{total} seeds] "
                    f"samples={total_samples} (~{avg:.1f}/seed) failed={failed_count} "
                    f"({rate:.1f} seeds/s, ETA {eta/60:.0f}m)"
                )

    return {
        "total_seeds": total,
        "total_samples": total_samples,
        "failed_seeds": failed_count,
        "elapsed_seconds": time.time() - start_time,
    }


def test(
    tools_file: str,
    seeds_file: str,
    model: str = DEFAULT_MODEL,
    timeout: int = SEED_TIMEOUT_SECONDS,
    cache_dir: str = ".cache/tool_db",
):
    """Run pipeline on a single random seed (smoke test)."""
    all_tools = _load_tools(tools_file)
    pipeline = build_pipeline(all_tools=all_tools, model=model, cache_dir=cache_dir)

    with open(seeds_file) as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    seed = random.choice(seeds)
    print(f"Seed tool: {seed['tool_name']}")
    print("Running pipeline...")

    import traceback
    try:
        initial_state = make_initial_state(seed)
        result = asyncio.run(
            asyncio.wait_for(
                asyncio.to_thread(pipeline.invoke, initial_state),
                timeout=timeout,
            )
        )
    except asyncio.TimeoutError:
        print(f"\n=== TIMEOUT after {timeout}s ===")
        raise
    except Exception:
        print("\n=== PIPELINE ERROR ===")
        traceback.print_exc()
        raise

    samples = list(result.get("completed_samples", []))
    print(f"\n=== RESULTS ===")
    print(f"Completed samples: {len(samples)}")
    for s in samples:
        print(f"\n[{s['complexity'].upper()}]")
        print(f"  Query:  {s['query']}")
        print(f"  Answer: {s['answer']}")


def generate(
    tools_file: str,
    seeds_file: str,
    output_dir: str = "output",
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
    timeout: int = SEED_TIMEOUT_SECONDS,
    cache_dir: str = ".cache/tool_db",
):
    """Run full batch generation with async concurrency."""
    all_tools = _load_tools(tools_file)
    pipeline = build_pipeline(all_tools=all_tools, model=model, cache_dir=cache_dir)

    with open(seeds_file) as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Seeds file:  {seeds_file}")
    print(f"Output:      {output_dir}/")
    print(f"Seeds:       {len(seeds)}")
    print(f"Model:       {model}")
    print(f"Concurrency: {concurrency}")
    print(f"Timeout:     {timeout}s/seed")
    print()

    stats = asyncio.run(
        _run_generation_async(
            pipeline=pipeline, seeds=seeds, output_path=output_path,
            concurrency=concurrency, timeout=timeout,
        )
    )

    print(f"\nGeneration complete ({stats['elapsed_seconds']:.0f}s):")
    print(f"  Total seeds:   {stats['total_seeds']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Failed seeds:  {stats['failed_seeds']}")
    if stats["total_seeds"] > 0:
        per_seed = stats["total_samples"] / max(1, stats["total_seeds"] - stats["failed_seeds"])
        print(f"  Samples/seed:  {per_seed:.1f}")

    samples_path = output_path / "samples.jsonl"
    if samples_path.exists():
        with open(samples_path) as f:
            first = f.readline().strip()
        if first:
            s = json.loads(first)
            print(f"\n--- Sample preview ---")
            print(f"  Tier:   {s.get('complexity')}")
            print(f"  Query:  {s.get('query')}")
            print(f"  Answer: {s.get('answer')}")


def _add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--tools-file", required=True, help="Input tools JSONL file")
    p.add_argument("--seeds-file", required=True, help="Seeds JSONL file from seed_preparation.py")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"LiteLLM model string (default: {DEFAULT_MODEL})")
    p.add_argument("--timeout", type=int, default=SEED_TIMEOUT_SECONDS)
    p.add_argument("--cache-dir", default=".cache/tool_db", help="chromadb cache directory")


def main():
    parser = argparse.ArgumentParser(description="Realistic benchmark generation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="Single-seed smoke test")
    _add_common_args(test_parser)

    gen_parser = subparsers.add_parser("generate", help="Full batch generation")
    _add_common_args(gen_parser)
    gen_parser.add_argument("--output-dir", default="output")
    gen_parser.add_argument("--concurrency", type=int, default=8)

    pp_parser = subparsers.add_parser("postprocess", help="Convert samples.jsonl to eval-ready JSONL")
    pp_parser.add_argument("--input", "-i", required=True)
    pp_parser.add_argument("--output", "-o", required=True)
    pp_parser.add_argument("--tools-file", required=True, help="Tool catalog JSONL")
    pp_parser.add_argument("--complexity", choices=["easy", "medium", "hard"], default=None)

    args = parser.parse_args()

    if args.command == "test":
        test(
            tools_file=args.tools_file, seeds_file=args.seeds_file,
            model=args.model, timeout=args.timeout, cache_dir=args.cache_dir,
        )
    elif args.command == "generate":
        generate(
            tools_file=args.tools_file, seeds_file=args.seeds_file,
            output_dir=args.output_dir, model=args.model,
            concurrency=args.concurrency, timeout=args.timeout, cache_dir=args.cache_dir,
        )
    elif args.command == "postprocess":
        from .postprocess import postprocess as _postprocess
        stats = _postprocess(
            input_path=Path(args.input),
            output_path=Path(args.output),
            tools_file=args.tools_file,
            complexity=args.complexity,
        )
        print(f"Written: {stats['written']} / {stats['total_read']} records → {args.output}")


if __name__ == "__main__":
    main()
