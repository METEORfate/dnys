from __future__ import annotations

import json
import re
from typing import Any


EMOTIONS = {
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
}
PROFILE_LABELS = {
    "personality_traits": {"extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual", "agreeable", "assertive", "emotionally_stable", "sensitive"},
    "interests": {"study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design"},
    "style": {"brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user"},
}
MEMORY_RE = re.compile(r"^mem_[0-9]{6}$")


def normalize_prediction(sample_id: str, raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    profile = raw.get("user_profile") if isinstance(raw.get("user_profile"), dict) else {}
    clean_profile: dict[str, list[str]] = {}
    for name, allowed in PROFILE_LABELS.items():
        values = profile.get(name)
        valid = isinstance(values, list) and all(isinstance(value, str) for value in values) and len(values) == len(set(values)) and set(values) <= allowed
        clean_profile[name] = values if valid else []
    memories = raw.get("memory_refs")
    valid_memories = isinstance(memories, list) and all(isinstance(value, str) and MEMORY_RE.fullmatch(value) for value in memories) and len(memories) == len(set(memories))
    return {
        "id": sample_id,
        "response_text": raw.get("response_text").strip() if isinstance(raw.get("response_text"), str) else "",
        "emotion_label": raw.get("emotion_label") if isinstance(raw.get("emotion_label"), str) and raw.get("emotion_label") in EMOTIONS else "",
        "user_profile": clean_profile,
        "memory_refs": memories if valid_memories else [],
    }


def extract_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S | re.I)
    candidate = fenced.group(1) if fenced else cleaned
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("模型输出中没有可解析的 JSON object")


def recover_response_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    match = re.search(r'"response_text"\s*:\s*"((?:\\.|[^"\\])*)', cleaned, flags=re.S)
    if match:
        try:
            value = json.loads(f'"{match.group(1)}"')
            if isinstance(value, str):
                return value.strip()
        except json.JSONDecodeError:
            pass
    return cleaned


def parse_prediction(sample_id: str, text: str) -> tuple[dict[str, Any], bool]:
    try:
        raw = extract_object(text)
        failed = False
    except ValueError:
        raw = {"response_text": recover_response_text(text), "emotion_label": "", "user_profile": {}, "memory_refs": []}
        failed = True
    return normalize_prediction(sample_id, raw), failed


def profile_complete(prediction: dict[str, Any]) -> bool:
    profile = prediction.get("user_profile")
    return isinstance(profile, dict) and all(isinstance(profile.get(name), list) for name in PROFILE_LABELS)


def memory_refs_valid(prediction: dict[str, Any]) -> bool:
    refs = prediction.get("memory_refs")
    return isinstance(refs, list) and all(isinstance(value, str) and MEMORY_RE.fullmatch(value) for value in refs) and len(refs) == len(set(refs))


def memory_refs_value_valid(refs: Any) -> bool:
    return isinstance(refs, list) and all(isinstance(value, str) and MEMORY_RE.fullmatch(value) for value in refs) and len(refs) == len(set(refs))
