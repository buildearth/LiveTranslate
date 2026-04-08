from langchain_core.runnables import RunnableConfig

from live_summary.models import StateTransition, TopicStatus
from live_summary.state import StreamState

DEFAULT_MAX_CONTEXT_ITEMS = 10


def apply_changes(state: StreamState, config: RunnableConfig | None = None) -> dict:
    """确定性状态变更：根据 StateTransition 更新状态"""
    transition = StateTransition(**state["_transition"])
    current_topic = state["current_topic"]
    new_messages = state.get("new_messages", [])

    # 获取最大上下文条目数
    max_context_items = DEFAULT_MAX_CONTEXT_ITEMS
    if config and "configurable" in config:
        max_context_items = config["configurable"].get(
            "max_context_items", max_context_items
        )

    updates: dict = {
        "overview": [transition.overview_delta],  # add reducer 追加
        "host_tips": [tip.model_dump() for tip in transition.host_tips],
        "_transition": None,
        "_topic_changed": transition.topic_changed,
    }

    if transition.topic_changed:
        # 归档当前话题
        last_timestamp = (
            new_messages[-1]["timestamp"]
            if new_messages
            else current_topic["start_time"]
        )
        archived_topic = {
            **current_topic,
            "status": TopicStatus.ARCHIVED.value,
            "summary": transition.current_topic_summary,
            "end_time": last_timestamp,
        }
        updates["archived_topics"] = [archived_topic]  # add reducer 会追加

        # 创建新话题
        new_counter = state["topic_counter"] + 1
        first_timestamp = (
            new_messages[0]["timestamp"] if new_messages else last_timestamp
        )
        updates["current_topic"] = {
            "id": f"topic_{new_counter:03d}",
            "title": transition.new_topic_title or "",
            "summary": transition.new_topic_summary or "",
            "status": TopicStatus.ACTIVE.value,
            "start_time": first_timestamp,
            "end_time": None,
        }
        updates["topic_counter"] = new_counter
    else:
        # 更新当前话题摘要
        updates["current_topic"] = {
            **current_topic,
            "summary": transition.current_topic_summary,
        }

    # 裁剪 recent_context 滑动窗口
    all_context = list(state.get("recent_context", []))
    if len(all_context) > max_context_items:
        updates["recent_context"] = all_context[-max_context_items:]

    return updates
