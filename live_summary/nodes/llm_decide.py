from typing import Callable

from langchain_core.language_models import BaseChatModel

from live_summary.models import StateTransition
from live_summary.prompts import TIP_ADVISORS, build_prompt
from live_summary.state import StreamState


def make_llm_decide(
    llm: BaseChatModel,
    tip_advisors: dict[str, str] | None = None,
) -> Callable[[StreamState], dict]:
    """工厂函数：创建 llm_decide 节点，闭包捕获 llm 和 tip_advisors"""
    structured_llm = llm.with_structured_output(StateTransition)
    advisors = tip_advisors or TIP_ADVISORS

    def llm_decide(state: StreamState) -> dict:
        prompt = build_prompt(state, advisors)
        transition: StateTransition = structured_llm.invoke(prompt)
        return {"_transition": transition.model_dump()}

    return llm_decide
