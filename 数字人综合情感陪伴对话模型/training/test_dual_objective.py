import unittest,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dual_objective_stream import DualObjectiveStream
from emotion_anchor_loss import emotion_labels
from transformers import AutoTokenizer
class DualObjectiveTests(unittest.TestCase):
 def test_rank_local_fixed_schedule(self):
  r=[{'source':i} for i in range(7334)];e=[{'source':i} for i in range(1200)]
  a=DualObjectiveStream(r,e,0,2);b=DualObjectiveStream(r,e,1,2);sa=list(a.schedule());sb=list(b.schedule())
  self.assertEqual(len(sa),1280);self.assertEqual([o for o,i in sa],['response','response','response','emotion']*320)
  self.assertTrue(all(i!=j and o==p for (o,i),(p,j) in zip(sa,sb)));self.assertEqual(sa,list(DualObjectiveStream(r,e,0,2).schedule()))
 def test_real_tokenizer_scalar_mask(self):
  R=Path(__file__).resolve().parents[1];tok=AutoTokenizer.from_pretrained(R/'participant/model/Qwen2___5-7B-Instruct',local_files_only=True,use_fast=True)
  for label in ['anxiety','mixed','relaxed','fear','helplessness','sadness']:
   target={'response_text':'包含伪字段 "emotion_label": "joy"，不能监督。','emotion_label':label,'user_profile':{'personality_traits':['sensitive'],'interests':[],'style':['direct']},'memory_refs':[]}
   messages=[{'role':'system','content':'test'},{'role':'user','content':'user: test'},{'role':'assistant','content':json.dumps(target,ensure_ascii=False)}]
   f,a=emotion_labels(tok,messages);selected=[v for v in f['labels'] if v!=-100];self.assertEqual(tok.decode(selected),label);self.assertEqual(a['non_emotion_supervised'],0)
if __name__=='__main__':unittest.main()
