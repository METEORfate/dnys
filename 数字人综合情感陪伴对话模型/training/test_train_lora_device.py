import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from .train_lora import build_model_kwargs, get_local_rank
except ImportError:
    from train_lora import build_model_kwargs, get_local_rank


class FakeCuda:
    def __init__(self):
        self.selected = None

    def set_device(self, rank):
        self.selected = rank


class FakeTorch:
    bfloat16 = object()


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TrainDevicePlacementTest(unittest.TestCase):
    def test_local_rank_from_environment(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            self.assertEqual(get_local_rank(), 0)
        with patch.dict(os.environ, {"LOCAL_RANK": "1"}):
            self.assertEqual(get_local_rank(), 1)

    def test_quantized_model_uses_rank_device_map(self):
        options = SimpleNamespace(load_in_4bit=True)
        stack = {"BitsAndBytesConfig": FakeBitsAndBytesConfig}
        kwargs = build_model_kwargs(FakeTorch, stack, options, 1)
        self.assertEqual(kwargs["device_map"], {"": 1})
        self.assertTrue(kwargs["quantization_config"].kwargs["load_in_4bit"])


if __name__ == "__main__":
    unittest.main()
