import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from style_postprocessor import StylePostprocessor
class ProfileTest(unittest.TestCase):
    def make(self):
        obj=StylePostprocessor.__new__(StylePostprocessor)
        obj.loaded=obj.personality_loaded=True
        obj.fallback_count=obj.invalid_output_count=obj.personality_fallback_count=obj.personality_override_count=0
        obj.predict_latencies_ms=[]
        obj.vectorizer=Mock()
        obj.model={"svc":Mock(),"classes":[["direct"]]}
        obj.model["svc"].predict.return_value=[0]
        obj.personality_model={"svc":Mock(),"classes":[["sensitive"]]}
        obj.personality_model["svc"].predict.return_value=[0]
        obj.personality_model["svc"].decision_function.return_value=np.array([[1.,.5,-1.]])
        return obj
    def original(self):
        return {"id":"test_000001_t1","response_text":"hello","emotion_label":"anxiety","memory_refs":[],"user_profile":{"personality_traits":[],"interests":[],"style":[]}}
    def test_single_transform_and_boundary(self):
        obj=self.make();b=self.original();f=obj.apply(b,[{"role":"user","content":"hi"}]);obj.vectorizer.transform.assert_called_once();self.assertEqual(f["user_profile"]["personality_traits"],["sensitive"]);self.assertEqual(b["user_profile"]["personality_traits"],[])
        for key in ["id","response_text","emotion_label","memory_refs"]:self.assertEqual(f[key],b[key])
    def test_below_boundary_preserves_personality(self):
        obj=self.make();obj.personality_model["svc"].decision_function.return_value=np.array([[1.,.501,-1.]])
        self.assertEqual(obj.apply(self.original(),[{"role":"user","content":"hi"}])["user_profile"]["personality_traits"],[])
    def test_personality_failure_keeps_style(self):
        obj=self.make();obj.personality_model["svc"].decision_function.side_effect=RuntimeError("injected")
        f=obj.apply(self.original(),[{"role":"user","content":"hi"}]);self.assertEqual(f["user_profile"]["style"],["direct"]);self.assertEqual(f["user_profile"]["personality_traits"],[]);self.assertEqual(obj.personality_fallback_count,1)
    def test_invalid_personality_keeps_style(self):
        obj=self.make();obj.personality_model["classes"]=[["invalid"]]
        f=obj.apply(self.original(),[{"role":"user","content":"hi"}]);self.assertEqual(f["user_profile"]["style"],["direct"]);self.assertEqual(f["user_profile"]["personality_traits"],[])
if __name__=="__main__":unittest.main()
