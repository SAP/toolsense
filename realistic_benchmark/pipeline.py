"""
Pipeline assembly for the realistic retrieval benchmark generator.

Outer graph (fan-out/fan-in):
  shared_context → [easy_subgraph, medium_subgraph, hard_subgraph] → aggregator

Each tier subgraph:
  generate → validate → (if done) exit
                      → (if retry) back to generate
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .environment import EvalGenEnv
from .nodes.aggregator import AggregatorNode
from .nodes.easy.generate import EasyGenerateNode
from .nodes.easy.validate import EasyValidateNode
from .nodes.hard.generate import HardGenerateNode
from .nodes.hard.validate import HardValidateNode
from .nodes.medium.generate import MediumGenerateNode
from .nodes.medium.validate import MediumValidateNode
from .nodes.shared_context import SharedContextNode, build_retriever


def _build_tier_pipeline(generate_node, validate_node, tier_name: str):
    def route_validation(state: dict) -> str:
        return "done" if state.get(f"{tier_name}_done", False) else "retry"

    g = StateGraph(EvalGenEnv)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_edge(START, "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", route_validation, {"done": END, "retry": "generate"})
    return g.compile()


def build_pipeline(
    all_tools: list[dict],
    model: str = "openai/gpt-4o",
    samples_per_tier: int = 3,
    embedding_model: str = "text-embedding-3-large",
    cache_dir: str = ".cache/tool_db",
):
    """Build and compile the generation pipeline.

    Args:
        all_tools: Full tool catalog (list of dicts with tool_name, tool_description).
        model: LiteLLM model string for generation and validation.
        samples_per_tier: Target number of samples per complexity tier per seed.
        embedding_model: OpenAI embedding model for hard-negative retrieval.
        cache_dir: Directory for persisting the chromadb tool index.
    """
    retriever = build_retriever(all_tools, cache_dir, embedding_model)

    shared_context = SharedContextNode(all_tools=all_tools, retriever=retriever)

    easy_pipeline = _build_tier_pipeline(
        generate_node=EasyGenerateNode(model=model, n_samples=samples_per_tier),
        validate_node=EasyValidateNode(model=model),
        tier_name="easy",
    )
    medium_pipeline = _build_tier_pipeline(
        generate_node=MediumGenerateNode(model=model, n_samples=samples_per_tier),
        validate_node=MediumValidateNode(model=model),
        tier_name="medium",
    )
    hard_pipeline = _build_tier_pipeline(
        generate_node=HardGenerateNode(model=model, n_samples=min(samples_per_tier, 2)),
        validate_node=HardValidateNode(model=model),
        tier_name="hard",
    )

    aggregator = AggregatorNode()

    outer = StateGraph(EvalGenEnv)
    outer.add_node("shared_context", shared_context)
    outer.add_node("easy_subgraph", easy_pipeline)
    outer.add_node("medium_subgraph", medium_pipeline)
    outer.add_node("hard_subgraph", hard_pipeline)
    outer.add_node("aggregator", aggregator)

    outer.add_edge(START, "shared_context")
    outer.add_edge("shared_context", "easy_subgraph")
    outer.add_edge("shared_context", "medium_subgraph")
    outer.add_edge("shared_context", "hard_subgraph")
    outer.add_edge(["easy_subgraph", "medium_subgraph", "hard_subgraph"], "aggregator")
    outer.add_edge("aggregator", END)

    return outer.compile()
