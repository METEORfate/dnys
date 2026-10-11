import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
START = (ROOT / "start.sh").read_text(encoding="utf-8")
CONFIG = json.loads((ROOT / "configs" / "inference_config.json").read_text(encoding="utf-8"))


class LoraRuntimeTest(unittest.TestCase):
    def test_vllm_lora_flags_and_official_path(self):
        self.assertIn("--enable-lora", START)
        self.assertRegex(START, r'--lora-modules\s+"emotion-companion-lora=\$ADAPTER_DIR"')
        self.assertIn('ADAPTER_DIR="$SCRIPT_DIR/model/lora_adapter"', START)

    def test_served_model_matches_config(self):
        self.assertEqual(CONFIG["api_model"], "emotion-companion-lora")
        self.assertEqual(CONFIG["model_path"], "./model/Qwen2___5-7B-Instruct")
        self.assertEqual(CONFIG["lora_adapter_path"], "./model/lora_adapter")

    def test_missing_adapter_is_explicit_failure(self):
        self.assertIn('[[ -f "$ADAPTER_DIR/adapter_config.json" && -f "$ADAPTER_DIR/adapter_model.safetensors" ]]', START)
        check = '[[ -f "$ADAPTER_DIR/adapter_config.json" && -f "$ADAPTER_DIR/adapter_model.safetensors" ]]'
        self.assertIn('exit 1', START[START.index(check):START.index(check) + 300])

    def test_participant_uses_shared_core(self):
        source = (ROOT / "run_inference.py").read_text(encoding="utf-8")
        self.assertIn("from core.message_builder import build_messages", source)
        self.assertIn("from core.output_schema import", source)
        self.assertIn("from core.prompt_loader import load_system_prompt", source)


if __name__ == "__main__":
    unittest.main()
