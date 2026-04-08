import logging
from typing import Callable

from live_summary.state import StreamState

logger = logging.getLogger(__name__)


def make_sync_graphiti(
    graphiti_uri: str | None = None,
) -> Callable[[StreamState], dict]:
    """工厂函数：创建 sync_graphiti 节点

    如果 graphiti_uri 为 None，返回一个 no-op 节点。
    """

    if graphiti_uri is None:
        def noop_sync(state: StreamState) -> dict:
            return {}
        return noop_sync

    async def sync_graphiti(state: StreamState) -> dict:
        """旁路节点：将对话写入知识图谱，提取实体和关系"""
        try:
            from graphiti_core import Graphiti
            from graphiti_core.nodes import EpisodeType

            graphiti = Graphiti(graphiti_uri, "neo4j", "password")
            dialogue_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in state.get("new_messages", [])
            )
            if dialogue_text:
                await graphiti.add_episode(
                    name=f"session_{state['session_id']}_topic_{state['current_topic']['id']}",
                    episode_body=dialogue_text,
                    source=EpisodeType.text,
                    source_description=f"直播对话 - {state['current_topic'].get('title', '')}",
                )
        except Exception:
            logger.warning("Graphiti sync failed, skipping", exc_info=True)
        return {}  # 不修改状态，纯旁路

    return sync_graphiti
