import unittest,json
from pathlib import Path
from transformers import AutoTokenizer
from response_only_loss import response_labels
class ResponseMaskTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=Path(__file__).resolve().parents[1];cls.tok=AutoTokenizer.from_pretrained(root/'participant/model/Qwen2___5-7B-Instruct',local_files_only=True,use_fast=True);
  with (root/'training/data/train_sft.jsonl').open() as handle:cls.messages=json.loads(handle.readline())['messages']
 def test_escaped_strings_and_structured_fields(self):
  for text in ['他说"你好"。\n欢迎！','路径 C:\\example，别担心。','好']:
   messages=[dict(x) for x in self.messages];messages[-1]['content']=json.dumps({'response_text':text,'emotion_label':'anxiety','user_profile':{'personality_traits':['sensitive'],'interests':['pets'],'style':['direct']},'memory_refs':['mem_000001']},ensure_ascii=False,separators=(',',':'))
   f,a=response_labels(self.tok,messages);self.assertGreater(a['response_tokens'],0);self.assertEqual(a['non_response_supervised'],0)
   self.assertEqual(sum(v!=-100 for v in f['labels']),a['response_tokens'])
 def test_empty_response_rejected(self):
  messages=[dict(x) for x in self.messages];messages[-1]['content']=json.dumps({'response_text':'','emotion_label':'neutral','user_profile':{},'memory_refs':[]})
  with self.assertRaises(ValueError):response_labels(self.tok,messages)
if __name__=='__main__':unittest.main()
