from pathlib import Path
from copy import deepcopy
import time
from core.output_schema import PROFILE_LABELS

class StylePostprocessor:
    def __init__(self):
        self.fallback_count = 0
        self.invalid_output_count = 0
        self.personality_override_count = 0
        self.personality_fallback_count = 0
        self.predict_latencies_ms = []
        self.loaded = False
        self.personality_loaded = False
        try:
            import joblib
            root = Path(__file__).resolve().parent / "profile_classifier"
            self.vectorizer = joblib.load(root / "vectorizer.joblib")
            models = joblib.load(root / "classifiers.joblib")
            self.model = models["style"]
            if self.model["winner"] != "EXACT_SET":
                raise ValueError("fixed strategy mismatch")
            self.loaded = True
            try:
                self.personality_model = models["personality_traits"]
                if not callable(getattr(self.personality_model["svc"], "decision_function", None)):
                    raise ValueError("personality decision_function unavailable")
                self.personality_loaded = True
            except Exception as exc:
                print("PERSONALITY_CLASSIFIER_LOAD_ERROR:", type(exc).__name__, str(exc), flush=True)
        except Exception as exc:
            print("STYLE_CLASSIFIER_LOAD_ERROR:", type(exc).__name__, str(exc), flush=True)

    @staticmethod
    def user_history(history):
        texts = [m["content"].strip() for m in history if m.get("role") == "user"]
        if not texts:
            raise ValueError("no user history")
        return "\n\n".join(("[USER_LATEST]" if i == len(texts)-1 else "[USER]") + "\n" + s for i, s in enumerate(texts))

    @staticmethod
    def valid_set(value, dimension):
        if value == "__EMPTY__" or value == ("__EMPTY__",):
            value = []
        if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) or v not in PROFILE_LABELS[dimension] for v in value):
            raise ValueError("illegal " + dimension)
        return sorted(set(value))

    def apply(self, result, history):
        started = time.perf_counter()
        try:
            if not self.loaded:
                raise RuntimeError("classifier unavailable")
            text = self.user_history(history)
            X = self.vectorizer.transform([text])
        except Exception as exc:
            self.fallback_count += 1
            self.personality_fallback_count += 1
            print("PROFILE_CLASSIFIER_FALLBACK:", type(exc).__name__, str(exc), flush=True)
            return result
        updated = deepcopy(result)
        try:
            if not isinstance(result.get("user_profile"), dict) or not isinstance(result["user_profile"].get("style"), list):
                raise ValueError("invalid original schema")
            index = self.model["svc"].predict(X)[0]
            value = self.valid_set(self.model["classes"][index], "style")
            updated["user_profile"]["style"] = value
        except Exception as exc:
            self.fallback_count += 1
            if isinstance(exc, ValueError) and str(exc).startswith("illegal"):
                self.invalid_output_count += 1
            print("STYLE_CLASSIFIER_FALLBACK:", type(exc).__name__, str(exc), flush=True)
        try:
            if not self.personality_loaded:
                raise RuntimeError("personality classifier unavailable")
            original = result["user_profile"]["personality_traits"]
            if not isinstance(original, list):
                raise ValueError("invalid original personality schema")
            model = self.personality_model
            index = model["svc"].predict(X)[0]
            value = self.valid_set(model["classes"][index], "personality_traits")
            scores = model["svc"].decision_function(X)
            import numpy as np
            scores = np.asarray(scores)
            if scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] < 2 or not np.isfinite(scores).all():
                raise ValueError("invalid personality scores")
            ordered = np.sort(scores[0])
            margin = float(ordered[-1] - ordered[-2])
            if set(value) != set(original) and margin >= 0.50:
                updated["user_profile"]["personality_traits"] = value
                self.personality_override_count += 1
        except Exception as exc:
            self.personality_fallback_count += 1
            print("PERSONALITY_CLASSIFIER_FALLBACK:", type(exc).__name__, str(exc), flush=True)
        self.predict_latencies_ms.append((time.perf_counter() - started) * 1000)
        return updated
