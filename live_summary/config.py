from pydantic import BaseModel


class SummarizerConfig(BaseModel):
    """总结器配置"""

    recent_context_max_items: int = 10  # 滑动窗口保留的上下文段数
    language: str = "zh"  # 输出语言
    prompt_template_path: str | None = None  # 自定义 prompt 模板路径
    graphiti_uri: str | None = None  # Graphiti Neo4j 连接地址，None 则禁用
    tip_advisors: dict[str, str] | None = None  # 自定义指导方向，None 用默认
