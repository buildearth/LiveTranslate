from enum import Enum

from pydantic import BaseModel, Field


class Message(BaseModel):
    """直播对话消息"""

    role: str  # 说话人标识，如 "主播", "嘉宾A"
    content: str  # 对话内容
    timestamp: float  # Unix 时间戳（秒）


class SummaryInput(BaseModel):
    """总结请求"""

    session_id: str  # 直播场次 ID
    messages: list[Message]  # 本批新对话（按时间排序）


class TopicStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class Topic(BaseModel):
    """话题结构"""

    id: str  # 话题唯一 ID，如 "topic_001"
    title: str  # 话题标题
    summary: str  # 该话题的摘要
    status: TopicStatus
    start_time: float  # 话题开始时间戳
    end_time: float | None = None  # 话题结束时间戳，ACTIVE 时为 None


class HostTip(BaseModel):
    """主播指导建议"""

    category: str  # 指导类型标识
    content: str  # 具体建议
    priority: str = "medium"  # "high" | "medium" | "low"


class StateTransition(BaseModel):
    """LLM 输出的状态转移指令"""

    topic_changed: bool  # 是否发生话题切换
    current_topic_summary: str  # 更新后的当前话题摘要（增量融合）
    new_topic_title: str | None = None  # 若切换，新话题标题
    new_topic_summary: str | None = None  # 若切换，新话题初始摘要
    overview_delta: str  # 本轮概述增量（追加到已有概述后）
    host_tips: list[HostTip] = Field(default_factory=list)


class SummaryMeta(BaseModel):
    """本轮处理的元信息"""

    topic_changed: bool  # 本轮是否发生话题切换
    total_topics: int  # 累计话题数（含当前）
    duration_seconds: float  # 直播已进行时长（秒）


class SummaryOutput(BaseModel):
    """对外输出结构"""

    session_id: str
    overview: str  # 全场概述（1-3 句话）
    current_topic: Topic  # 当前活跃话题
    archived_topics: list[Topic]  # 已归档话题（按时间排序）
    host_tips: list[HostTip]  # 主播指导建议
    meta: SummaryMeta  # 本轮元信息
