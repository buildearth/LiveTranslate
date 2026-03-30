# analysis_presets.py
"""Analysis scene presets and structured prompt template assembly."""

from dataclasses import dataclass, field

# Tags available for structured template editor
FOCUS_TAGS = [
    "情绪变化", "关键诉求", "矛盾点", "报价", "承诺",
    "让步信号", "关键信息", "未回答问题", "互动节奏",
]

OUTPUT_TAGS = [
    "局势判断", "建议话术", "风险提醒", "价格对比",
    "情绪分析", "问题归类", "话题建议", "信息提取",
]


@dataclass
class AnalysisPreset:
    name: str
    role: str = ""
    focus_tags: list[str] = field(default_factory=list)
    output_tags: list[str] = field(default_factory=list)
    extra_instructions: str = ""
    is_advanced: bool = False
    advanced_prompt: str = ""
    builtin: bool = False  # True for built-in presets (not editable)

    def build_prompt(self) -> str:
        """Assemble a full system prompt from structured fields or advanced text."""
        if self.is_advanced and self.advanced_prompt:
            return self.advanced_prompt
        parts = []
        if self.role:
            parts.append(f"你是一位{self.role}。")
        else:
            parts.append("你是一位专业的直播对话分析助手。")
        parts.append("根据对话摘要和最新对话内容，给出实时分析和建议。")
        if self.focus_tags:
            parts.append(f"\n重点关注：{', '.join(self.focus_tags)}。")
        if self.output_tags:
            parts.append(f"\n输出需包含：{', '.join(self.output_tags)}。")
        if self.extra_instructions:
            parts.append(f"\n{self.extra_instructions}")
        parts.append("\n请用简洁的结构化格式输出（使用 ## 标题分段），便于快速阅读。")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "focus_tags": self.focus_tags,
            "output_tags": self.output_tags,
            "extra_instructions": self.extra_instructions,
            "is_advanced": self.is_advanced,
            "advanced_prompt": self.advanced_prompt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisPreset":
        return cls(
            name=d.get("name", ""),
            role=d.get("role", ""),
            focus_tags=d.get("focus_tags", []),
            output_tags=d.get("output_tags", []),
            extra_instructions=d.get("extra_instructions", ""),
            is_advanced=d.get("is_advanced", False),
            advanced_prompt=d.get("advanced_prompt", ""),
        )


# Built-in presets
ANALYSIS_PRESETS: dict[str, AnalysisPreset] = {
    "带货直播": AnalysisPreset(
        name="带货直播",
        role="直播带货分析师",
        focus_tags=["报价", "承诺", "让步信号", "关键诉求"],
        output_tags=["价格对比", "建议话术", "风险提醒"],
        extra_instructions="注意识别对方的定价策略和限时话术，提醒砍价机会。",
        builtin=True,
    ),
    "商务谈判": AnalysisPreset(
        name="商务谈判",
        role="商务谈判顾问",
        focus_tags=["关键诉求", "矛盾点", "让步信号", "承诺"],
        output_tags=["局势判断", "建议话术", "风险提醒"],
        extra_instructions="分析双方立场差异，识别对方的底线信号和让步空间。",
        builtin=True,
    ),
    "情感连线": AnalysisPreset(
        name="情感连线",
        role="情感分析师",
        focus_tags=["情绪变化", "关键诉求", "矛盾点"],
        output_tags=["情绪分析", "建议话术", "风险提醒"],
        extra_instructions="注意识别对方的情绪转折点和语气变化，提供共情话术建议。",
        builtin=True,
    ),
    "采访访谈": AnalysisPreset(
        name="采访访谈",
        role="访谈助理",
        focus_tags=["关键信息", "未回答问题", "关键诉求"],
        output_tags=["信息提取", "话题建议", "建议话术"],
        extra_instructions="提取对方回答中的关键事实，标记被回避或未完整回答的问题。",
        builtin=True,
    ),
    "娱乐连麦": AnalysisPreset(
        name="娱乐连麦",
        role="直播互动策划",
        focus_tags=["互动节奏", "情绪变化"],
        output_tags=["话题建议", "建议话术"],
        extra_instructions="关注对话节奏和气氛变化，在冷场时及时建议新话题。",
        builtin=True,
    ),
    "客服售后": AnalysisPreset(
        name="客服售后",
        role="客服督导",
        focus_tags=["关键诉求", "情绪变化", "矛盾点"],
        output_tags=["问题归类", "建议话术", "风险提醒"],
        extra_instructions="识别客户核心问题和情绪状态，判断是否需要升级处理。",
        builtin=True,
    ),
}

SUMMARY_COMPRESS_PROMPT = (
    "将以下对话摘要和新增对话合并，生成简洁的结构化摘要。\n"
    "保留：关键事实、双方立场、已达成共识、待解决问题、情绪变化。\n"
    "删除：重复信息、无实质内容的寒暄。\n"
    "输出纯文本，不超过500字。"
)
