"""seed_preparation.py — Stratified sampling of seeds from a tool catalog.

Tool names following the format "ServiceName&&MethodName" use the service prefix
as the domain for stratified sampling, ensuring broad domain coverage.

Sampling strategy:
- Group all tools by service (domain)
- Proportionally allocate N seeds across domains (min 1 per domain)
- Within each domain, sample randomly without replacement

Usage:
    python -m realistic_benchmark.seed_preparation \\
        --tools-file tools.jsonl \\
        --n-seeds 1000 \\
        --output seeds/seeds.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_N_SEEDS = 1000


def _extract_domain(tool: dict) -> str:
    """Extract service/domain from a tool name ('Service&&Method' → 'Service')."""
    name = tool.get("tool_name", "")
    return name.split("&&")[0] if "&&" in name else name


def stratified_sample(all_tools: list[dict], n_seeds: int, seed: int = 42) -> list[dict]:
    """Proportionally sample n_seeds tools across domains (services).

    Each domain gets at least 1 seed. Allocation is proportional to domain size,
    with remainders distributed to the largest domains first.
    """
    rng = random.Random(seed)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for tool in all_tools:
        by_domain[_extract_domain(tool)].append(tool)

    domains = sorted(by_domain.keys())
    n_domains = len(domains)
    effective_n = min(n_seeds, len(all_tools))

    if n_domains >= effective_n:
        sampled_domains = rng.sample(domains, effective_n)
        return [rng.choice(by_domain[d]) for d in sampled_domains]

    total = len(all_tools)
    raw_alloc = {d: max(1, math.floor(len(by_domain[d]) / total * effective_n)) for d in domains}

    allocated = sum(raw_alloc.values())
    remainder = effective_n - allocated
    if remainder > 0:
        fracs = {
            d: (len(by_domain[d]) / total * effective_n) - raw_alloc[d]
            for d in domains
        }
        top_domains = sorted(fracs, key=lambda d: -fracs[d])[:remainder]
        for d in top_domains:
            raw_alloc[d] += 1
    elif remainder < 0:
        small_domains = sorted(raw_alloc, key=lambda d: raw_alloc[d])
        for d in small_domains:
            if remainder == 0:
                break
            if raw_alloc[d] > 1:
                raw_alloc[d] -= 1
                remainder += 1

    sampled = []
    for d in domains:
        k = min(raw_alloc[d], len(by_domain[d]))
        sampled.extend(rng.sample(by_domain[d], k))

    rng.shuffle(sampled)
    return sampled[:effective_n]


def prepare_seeds(all_tools: list[dict], output_file: Path, n_seeds: int = DEFAULT_N_SEEDS) -> dict:
    """Sample seeds and write to JSONL. Returns summary stats."""
    seeds = stratified_sample(all_tools, n_seeds=n_seeds)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for tool in seeds:
            f.write(json.dumps(tool) + "\n")

    domain_counts: dict[str, int] = defaultdict(int)
    for tool in seeds:
        domain_counts[_extract_domain(tool)] += 1

    return {
        "total_tools": len(all_tools),
        "n_domains": len(domain_counts),
        "n_seeds": len(seeds),
        "domain_counts": dict(domain_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Stratified seed sampling for benchmark generation")
    parser.add_argument("--tools-file", required=True, help="Input tools JSONL file")
    parser.add_argument(
        "--n-seeds", type=int, default=DEFAULT_N_SEEDS,
        help=f"Number of seeds to sample (default: {DEFAULT_N_SEEDS})"
    )
    parser.add_argument("--output", required=True, help="Output JSONL path for seed records")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    with open(args.tools_file) as f:
        all_tools = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(all_tools)} tools from {args.tools_file}")

    output_file = Path(args.output)
    stats = prepare_seeds(all_tools=all_tools, output_file=output_file, n_seeds=args.n_seeds)

    print(f"\nStratified sampling: {stats['n_seeds']} seeds from {stats['n_domains']} domains")
    print(f"Written to: {output_file}")

    sorted_domains = sorted(stats["domain_counts"].items(), key=lambda x: -x[1])
    print(f"\nTop domains by seed count:")
    for domain, count in sorted_domains[:20]:
        pct = count / stats["n_seeds"] * 100
        print(f"  {domain}: {count} ({pct:.1f}%)")
    if len(sorted_domains) > 20:
        print(f"  ... and {len(sorted_domains) - 20} more domains")

    print(f"\nDone. Run generation with:")
    print(f"  python -m realistic_benchmark.run_generation generate \\")
    print(f"      --seeds-file {output_file} \\")
    print(f"      --tools-file {args.tools_file}")


if __name__ == "__main__":
    main()
