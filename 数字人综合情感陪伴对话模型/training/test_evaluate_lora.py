import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from .evaluate_lora import (
        attach_adapter_if_requested,
        normalize,
        parse_args,
        validate_paths,
    )
except ImportError:
    from evaluate_lora import attach_adapter_if_requested, normalize, parse_args, validate_paths


class FakePeftModel:
    calls = []

    @classmethod
    def from_pretrained(cls, base_model, adapter_path, is_trainable):
        cls.calls.append((base_model, adapter_path, is_trainable))
        return {"base": base_model, "adapter": adapter_path}


class EvaluateLoraModeTest(unittest.TestCase):
    def test_default_cli_is_base_only(self):
        with patch.object(sys, "argv", ["evaluate_lora.py"]):
            options = parse_args()
        self.assertIsNone(options.adapter_path)

    def test_base_only_does_not_attach_peft(self):
        base = object()
        model, mode = attach_adapter_if_requested(base, None, FakePeftModel)
        self.assertIs(model, base)
        self.assertEqual(mode, "base")
        self.assertEqual(FakePeftModel.calls, [])

    def test_lora_mode_attaches_adapter(self):
        FakePeftModel.calls.clear()
        base = object()
        adapter = Path("adapter")
        model, mode = attach_adapter_if_requested(base, adapter, FakePeftModel)
        self.assertEqual(mode, "lora")
        self.assertEqual(model["adapter"], adapter)
        self.assertEqual(FakePeftModel.calls, [(base, adapter, False)])

    def test_only_lora_mode_requires_adapter_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            data = root / "val.jsonl"
            model.mkdir()
            data.write_text("{}\n", encoding="utf-8")
            validate_paths(model, data, None)
            with self.assertRaises(FileNotFoundError):
                validate_paths(model, data, root / "missing-adapter")

    def test_memory_refs_string_list(self):
        result, _, valid = normalize({"memory_refs": ["mem_000001", "mem_000002"]})
        self.assertTrue(valid)
        self.assertEqual(result["memory_refs"], ["mem_000001", "mem_000002"])

    def test_memory_refs_dict_list(self):
        refs = [{"id": "xxx", "content": "xxx"}, {"content": "yyy", "id": "yyy"}]
        result, _, valid = normalize({"memory_refs": refs})
        self.assertFalse(valid)
        self.assertEqual(result["memory_refs"], [])

    def test_memory_refs_empty_list(self):
        result, _, valid = normalize({"memory_refs": []})
        self.assertTrue(valid)
        self.assertEqual(result["memory_refs"], [])

    def test_memory_refs_none(self):
        result, _, valid = normalize({"memory_refs": None})
        self.assertFalse(valid)
        self.assertEqual(result["memory_refs"], [])


if __name__ == "__main__":
    unittest.main()
