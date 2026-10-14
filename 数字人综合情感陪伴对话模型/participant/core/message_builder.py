from __future__ import annotations

from typing import Any, Iterable

from .prompt_loader import load_system_prompt


def conversation_text(history: Iterable[dict[str, Any]], sample_id: str | None = None) -> str:
    rendered: list[str] = []
    for turn in history:
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"} or not isinstance(turn.get("content"), str):
            raise ValueError("history turn must contain a user/assistant role and string content")
        rendered.append(f"{turn['role']}: {turn['content']}")
    prefix = f"样本 ID: {sample_id}\n" if sample_id is not None else ""
    return prefix + "对话历史:\n" + "\n".join(rendered)


def build_messages(sample: dict[str, Any], include_sample_id: bool = True) -> list[dict[str, str]]:
    history = sample.get("history")
    if not isinstance(history, list):
        raise ValueError("sample history must be a list")
    sample_id = sample.get("id") if include_sample_id else None
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": conversation_text(history, sample_id)},
    ]


def build_messages_from_turns(turns: Iterable[dict[str, Any]], sample_id: str | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": conversation_text(turns, sample_id)},
    ]


def build_messages_from_user_content(user_content: str) -> list[dict[str, str]]:
    if not isinstance(user_content, str):
        raise ValueError("user content must be a string")
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": user_content},
    ]
