from __future__ import annotations
import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "configs" / "system_prompt_v1.txt"
PROMPT_DIR = PROJECT_ROOT / "configs" / "prompts"

def resolve_prompt_path(path: str | Path | None = None) -> Path:
    selector = path if path is not None else os.environ.get("DNYS_PROMPT")
    if selector is None or str(selector).strip() == "":
        return PROMPT_PATH
    selector = str(selector).strip()
    named = PROMPT_DIR / (selector if selector.endswith(".txt") else f"prompt_{selector.lower()}.txt")
    if named.is_file():
        return named
    candidate = Path(selector)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Prompt not found: {selector}")

def load_system_prompt(path: str | Path | None = None) -> str:
    resolved = resolve_prompt_path(path)
    prompt = resolved.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"system prompt is empty: {resolved}")
    return prompt

def prompt_metadata(path: str | Path | None = None) -> dict:
    resolved = resolve_prompt_path(path)
    return {"prompt_path": str(resolved), "prompt_version": resolved.stem, "prompt_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}
