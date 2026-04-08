from live_summary.state import StreamState

# 指导方向注册表：category → prompt 描述
TIP_ADVISORS: dict[str, str] = {
    "topic": "基于已聊话题和直播主题，建议下一步聊什么、当前话题是否该收",
    "content": "检查当前话题遗漏的观众关心点",
    "pacing": "基于时长和互动频率，给出节奏提醒",
    # 扩展示例（按需启用）
    # "emotion": "感知观众情绪变化，建议调整语气或话题方向",
    # "selling": "识别带货转化时机，提醒上链接或强调卖点",
    # "risk": "检测敏感内容或违规风险，及时提醒",
}


def _build_advisor_section(tip_advisors: dict[str, str]) -> str:
    lines = []
    for category, description in tip_advisors.items():
        lines.append(f"- **{category}**: {description}")
    return "\n".join(lines)


def _build_archived_section(archived_topics: list[dict]) -> str:
    if not archived_topics:
        return "无"
    lines = []
    for t in archived_topics:
        title = t.get("title") or t.get("id", "未知话题")
        summary = t.get("summary", "")
        lines.append(f"- {title}: {summary}")
    return "\n".join(lines)


def build_prompt(state: StreamState, tip_advisors: dict[str, str] | None = None) -> str:
    if tip_advisors is None:
        tip_advisors = TIP_ADVISORS

    current_topic = state["current_topic"]
    recent_context = "\n".join(state.get("recent_context", []))
    overview_parts = state.get("overview", [])
    if isinstance(overview_parts, list):
        overview = "\n".join(overview_parts)
    else:
        overview = overview_parts or ""
    archived_section = _build_archived_section(state.get("archived_topics", []))
    advisor_section = _build_advisor_section(tip_advisors)

    return f"""你是一个直播内容分析助手。你的核心工作是**增量更新**直播状态——每次收到新的对话内容后，在已有总结的基础上融合新信息，产出更完整的总结。

## 增量更新原则

- **话题摘要是累积的**：当前话题摘要已经包含了该话题之前所有轮次的信息。你要做的是把新对话中的新信息融入现有摘要，而不是只总结新对话。输出的 current_topic_summary 必须覆盖该话题从开始到现在的全部要点。
- **全场概述是追加式的**：overview_delta 输出本轮新增的概述内容（2-4 句话），系统会自动追加到已有概述后面。不要重复已有概述中的内容。overview_delta 要包含本轮的**关键结论、核心数据、重要观点**，而不仅仅是"主播开始介绍XX"这样的流水账。
- **指导建议基于累积总结**：主播指导要基于完整的累积总结（已有概述 + 所有话题摘要）来生成，而不是仅基于最新一轮对话。

## 当前直播状态（你需要在此基础上增量更新）

全场概述：{overview or "（直播刚开始）"}

已归档话题（已完结，仅供参考）：
{archived_section}

当前话题：{current_topic.get("title") or "（尚未确定，请根据内容确定标题）"}
当前话题累积摘要：{current_topic.get("summary") or "（首次输入，请生成初始摘要）"}

## 新的对话内容（在上述状态基础上融合这些新信息）

{recent_context}

## 你需要完成的任务

1. **判断话题是否切换**：新对话是否开始了一个新话题？如果主播明确切换或内容明显偏离当前话题，则 topic_changed=true。

2. **增量更新话题摘要**：
   - 如果话题未切换：将新对话中的新信息融入现有「当前话题累积摘要」，输出更完整的 current_topic_summary。必须保留旧摘要中的关键信息，同时加入新内容。
   - 如果话题切换：current_topic_summary 是旧话题的最终摘要（融入最后一批对话后的完整版），new_topic_title 和 new_topic_summary 是新话题的初始摘要。

3. **追加全场概述**：overview_delta 写本轮新增的核心内容（2-4 句话），不要重复已有概述。要求：
   - 必须包含本轮出现的**关键信息**（核心论点、具体案例、关键数据等）
   - 必须包含本轮的**核心观点或结论**（建议、判断、共识等）
   - 不要写"主播开始介绍XX"这样的流水账，要写实质内容
   - 示例（好）："主播与嘉宾深入探讨异地恋信任问题，指出信任缺失多源于焦虑型依恋人格而非对方过错。给出核心建议：约定固定沟通时间（如每晚九点视频半小时），让彼此拥有确定的专属时间，嘉宾亲测有效。观众弹幕反馈共鸣强烈。"
   - 示例（差）："主播开始讨论异地恋信任问题。"

4. **基于累积总结生成主播指导**：基于完整的累积总结（全场概述 + 所有话题摘要），为主播提供建议。每条建议包含 category、content、priority（high/medium/low）。

### 指导方向
{advisor_section}

## 输出要求

严格按照 JSON schema 返回结果，不要添加额外内容。"""
