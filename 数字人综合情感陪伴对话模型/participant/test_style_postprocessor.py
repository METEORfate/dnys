import sys
import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_postprocessor import StylePostprocessor

class StylePostprocessorTest(unittest.TestCase):
    def make_processor(self, value):
        obj = StylePostprocessor.__new__(StylePostprocessor)
        obj.loaded = True
        obj.personality_loaded = False
        obj.personality_fallback_count = obj.personality_override_count = 0
        obj.fallback_count = obj.invalid_output_count = 0
        obj.predict_latencies_ms = []
        obj.vectorizer = Mock()
        obj.model = {"svc": Mock(), "classes": [value]}
        obj.model["svc"].predict.return_value = [0]
        return obj
    def test_only_style_changes(self):
        original = {"id": "test_000001_t1", "emotion_label": "anxiety", "response_text": "reply", "memory_refs": [], "user_profile": {"style": [], "personality_traits": ["sensitive"], "interests": []}}
        saved = deepcopy(original)
        obj = self.make_processor(["direct", "direct"])
        final = obj.apply(original, [{"role": "user", "content": "hello"}])
        self.assertEqual(original, saved)
        self.assertEqual(final["user_profile"]["style"], ["direct"])
        final["user_profile"]["style"] = saved["user_profile"]["style"]
        self.assertEqual(final, saved)
    def test_empty_set(self):
        obj = self.make_processor("__EMPTY__")
        result = obj.apply({"user_profile": {"style": ["direct"]}}, [{"role": "user", "content": "hello"}])
        self.assertEqual(result["user_profile"]["style"], [])
    def test_invalid_output_preserves_original(self):
        obj = self.make_processor(["illegal"])
        original = {"user_profile": {"style": ["direct"]}}
        self.assertEqual(obj.apply(original, [{"role": "user", "content": "hello"}]), original)
        self.assertEqual(obj.fallback_count, 1)
        self.assertEqual(obj.invalid_output_count, 1)
    def test_prediction_exception_preserves_original(self):
        obj = self.make_processor([])
        obj.model["svc"].predict.side_effect = RuntimeError("test failure")
        original = {"user_profile": {"style": []}}
        self.assertEqual(obj.apply(original, [{"role": "user", "content": "hello"}]), original)
    def test_user_history_excludes_other_roles(self):
        self.assertEqual(StylePostprocessor.user_history([{"role": "system", "content": "hidden"}, {"role": "user", "content": "first"}, {"role": "assistant", "content": "hidden"}, {"role": "user", "content": "last"}]), "[USER]\nfirst\n\n[USER_LATEST]\nlast")
if __name__ == "__main__":
    unittest.main()
