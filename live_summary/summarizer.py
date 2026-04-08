from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from live_summary.config import SummarizerConfig
from live_summary.graph import build_graph
from live_summary.models import (
    HostTip,
    Message,
    SummaryMeta,
    SummaryOutput,
    Topic,
    TopicStatus,
)


class LiveSummarizer:
    """增量直播总结器

    对外接口：传入 session_id 和新消息，返回 SummaryOutput。
    内部驱动 LangGraph 状态机，通过 thread_id 自动管理状态持久化。
    """

    def __init__(
        self,
        llm: BaseChatModel,
        config: SummarizerConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        extra_instructions: str = "",
    ):
        self._config = config or SummarizerConfig()
        self._extra_instructions = extra_instructions
        self.graph = build_graph(llm, self._config, checkpointer, extra_instructions=extra_instructions)

    def summarize(self, session_id: str, messages: list[Message]) -> SummaryOutput:
        """核心方法：输入新对话，返回更新后的总结"""
        graph_config = {
            "configurable": {
                "thread_id": session_id,
                "max_context_items": self._config.recent_context_max_items,
            }
        }

        # 检查是否有已有状态（非首次调用）
        existing_state = self.graph.get_state(graph_config)
        is_first_call = existing_state.values == {}

        input_data: dict = {
            "new_messages": [m.model_dump() for m in messages],
        }

        if is_first_call:
            # 首次调用，设置初始状态
            first_ts = messages[0].timestamp if messages else 0.0
            input_data.update(
                {
                    "session_id": session_id,
                    "overview": [],
                    "current_topic": {
                        "id": "topic_001",
                        "title": "",
                        "summary": "",
                        "status": TopicStatus.ACTIVE.value,
                        "start_time": first_ts,
                        "end_time": None,
                    },
                    "archived_topics": [],
                    "recent_context": [],
                    "topic_counter": 1,
                    "host_tips": [],
                    "_transition": None,
                    "_topic_changed": False,
                }
            )

        state = self.graph.invoke(input_data, graph_config)
        return self._to_output(state, session_id)

    @staticmethod
    def _to_output(state: dict, session_id: str) -> SummaryOutput:
        """将内部 StreamState 投影为 SummaryOutput"""
        current_topic_dict = state["current_topic"]
        current_topic = Topic(**current_topic_dict)

        archived_topics = [Topic(**t) for t in state.get("archived_topics", [])]

        host_tips = [HostTip(**t) for t in state.get("host_tips", [])]

        # 计算 meta
        all_topics = archived_topics + [current_topic]
        total_topics = len(all_topics)

        topic_changed = state.get("_topic_changed", False)

        # 计算时长
        start_time = current_topic.start_time
        if archived_topics:
            start_time = min(t.start_time for t in all_topics)
        latest_time = current_topic.start_time
        new_messages = state.get("new_messages", [])
        if new_messages:
            latest_time = max(m["timestamp"] for m in new_messages)
        duration_seconds = max(0.0, latest_time - start_time)

        meta = SummaryMeta(
            topic_changed=topic_changed,
            total_topics=total_topics,
            duration_seconds=duration_seconds,
        )

        return SummaryOutput(
            session_id=session_id,
            overview="\n".join(state.get("overview", [])),
            current_topic=current_topic,
            archived_topics=archived_topics,
            host_tips=host_tips,
            meta=meta,
        )
