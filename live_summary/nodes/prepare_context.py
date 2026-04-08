from live_summary.state import StreamState


def prepare_context(state: StreamState) -> dict:
    """将 new_messages 格式化为文本，追加到 recent_context"""
    formatted_lines = []
    for msg in state.get("new_messages", []):
        formatted_lines.append(f"{msg['role']}: {msg['content']}")

    formatted_text = "\n".join(formatted_lines)
    if not formatted_text:
        return {}

    existing = list(state.get("recent_context", []))
    existing.append(formatted_text)
    return {"recent_context": existing}
