from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from live_summary.config import SummarizerConfig
from live_summary.nodes.apply_changes import apply_changes
from live_summary.nodes.llm_decide import make_llm_decide
from live_summary.nodes.prepare_context import prepare_context
from live_summary.nodes.sync_graphiti import make_sync_graphiti
from live_summary.state import StreamState


def build_graph(
    llm: BaseChatModel,
    config: SummarizerConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    extra_instructions: str = "",
) -> CompiledStateGraph:
    """构建 LangGraph 状态图"""
    config = config or SummarizerConfig()

    llm_decide = make_llm_decide(llm, tip_advisors=config.tip_advisors, extra_instructions=extra_instructions)
    sync_graphiti = make_sync_graphiti(graphiti_uri=config.graphiti_uri)

    builder = StateGraph(StreamState)
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("llm_decide", llm_decide)
    builder.add_node("apply_changes", apply_changes)
    builder.add_node("sync_graphiti", sync_graphiti)

    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "llm_decide")
    builder.add_edge("llm_decide", "apply_changes")
    builder.add_edge("apply_changes", "sync_graphiti")
    builder.add_edge("sync_graphiti", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)
