from pathlib import Path
from copy import deepcopy
import time
from core.output_schema import PROFILE_LABELS
class StylePostprocessor:
    def __init__(self):
        self.fallback_count=0
        self.invalid_output_count=0
        self.predict_latencies_ms=[]
        self.loaded=False
        try:
            import joblib
            root=Path(__file__).resolve().parent/'profile_classifier'
            self.vectorizer=joblib.load(root/'vectorizer.joblib')
            self.model=joblib.load(root/'classifiers.joblib')['style']
            if self.model['winner']!='EXACT_SET': raise ValueError('fixed strategy mismatch')
            self.loaded=True
        except Exception as exc:
            print('STYLE_CLASSIFIER_LOAD_ERROR:',type(exc).__name__,str(exc),flush=True)
    @staticmethod
    def user_history(history):
        texts=[m['content'].strip() for m in history if m.get('role')=='user']
        if not texts: raise ValueError('no user history')
        return '\n\n'.join(('[USER_LATEST]' if i==len(texts)-1 else '[USER]')+'\n'+s for i,s in enumerate(texts))
    def _invalid_style(self):
        self.invalid_output_count+=1
        raise ValueError("illegal style")
    def apply(self,result,history):
        try:
            if not self.loaded: raise RuntimeError('classifier unavailable')
            if not isinstance(result.get('user_profile'),dict) or not isinstance(result['user_profile'].get('style'),list): raise ValueError('invalid original schema')
            text=self.user_history(history)
            started=time.perf_counter()
            index=self.model['svc'].predict(self.vectorizer.transform([text]))[0]
            self.predict_latencies_ms.append((time.perf_counter()-started)*1000)
            value=self.model['classes'][index]
            if value=='__EMPTY__' or value==('__EMPTY__',): value=[]
            if not isinstance(value,(list,tuple)) or any(not isinstance(v,str) or v not in PROFILE_LABELS['style'] for v in value): self._invalid_style()
            value=sorted(set(value))
            updated=deepcopy(result)
            updated['user_profile']['style']=value
            return updated
        except Exception as exc:
            self.fallback_count+=1
            print('STYLE_CLASSIFIER_FALLBACK:',type(exc).__name__,str(exc),flush=True)
            return result
