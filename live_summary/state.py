from operator import add
from typing import Annotated, TypedDict


def replace(old, new):
    """覆盖语义的 reducer"""
    return new


class StreamState(TypedDict):
    """LangGraph 内部状态结构"""

    session_id: Annotated[str, replace]
    overview: Annotated[list[str], add]  # 全场概述（追加式，每轮增加一条）
    current_topic: Annotated[dict, replace]  # 当前活跃话题（Topic dict）
    archived_topics: Annotated[list[dict], add]  # 已归档话题（append 语义）
    recent_context: Annotated[list[str], replace]  # 最近原始对话（滑动窗口，需裁剪）
    topic_counter: Annotated[int, replace]  # 话题 ID 自增计数器
    new_messages: Annotated[list[dict], replace]  # 本批新输入（每步覆盖）
    host_tips: Annotated[list[dict], replace]  # 主播实时指导（每轮覆盖）
    _transition: Annotated[dict | None, replace]  # llm_decide → apply_changes
    _topic_changed: Annotated[bool, replace]  # apply_changes 写入，summarizer 读取
