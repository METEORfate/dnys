from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "configs" / "system_prompt_v1.txt"


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    """Load the frozen competition prompt from the repository config."""
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"system prompt is empty: {path}")
    return prompt
