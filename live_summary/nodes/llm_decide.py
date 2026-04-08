import json
import re
from typing import Callable

from langchain_core.language_models import BaseChatModel

from live_summary.models import StateTransition
from live_summary.prompts import TIP_ADVISORS, build_prompt
from live_summary.state import StreamState

# JSON schema instruction appended to prompt
_JSON_SCHEMA_INSTRUCTION = """

请严格按以下 JSON 格式返回，不要包含任何其他内容：

```json
{
  "topic_changed": false,
  "current_topic_summary": "更新后的当前话题完整摘要",
  "new_topic_title": null,
  "new_topic_summary": null,
  "overview_delta": "本轮新增的概述内容（2-4句话）",
  "host_tips": [
    {"category": "topic", "content": "建议内容", "priority": "medium"},
    {"category": "content", "content": "建议内容", "priority": "high"}
  ]
}
```

注意：只返回 JSON，不要用 markdown 代码块包裹，不要添加解释。"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Try to find JSON in markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Try direct parse
    return json.loads(text)


def make_llm_decide(
    llm: BaseChatModel,
    tip_advisors: dict[str, str] | None = None,
) -> Callable[[StreamState], dict]:
    """Factory: create llm_decide node, closure captures llm and tip_advisors."""
    advisors = tip_advisors or TIP_ADVISORS

    def llm_decide(state: StreamState) -> dict:
        prompt = build_prompt(state, advisors) + _JSON_SCHEMA_INSTRUCTION
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        data = _extract_json(raw)
        transition = StateTransition(**data)
        return {"_transition": transition.model_dump()}

    return llm_decide
