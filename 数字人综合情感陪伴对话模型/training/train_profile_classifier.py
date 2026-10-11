import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,re,time,collections,pathlib,numpy as np,joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));from core.output_schema import PROFILE_LABELS
A=R/'analysis/v2_p1';A.mkdir(exist_ok=True)
def load(p):return [json.loads(x) for x in open(p,encoding='utf8') if x.strip()]
def users(o):
 texts=[]
 for m in o['messages']:
  if m['role']!='user':continue
  c=m['content']
  if c.startswith('对话历史:'):
   matches=list(re.finditer(r'(?m)^(user|assistant): ',c))
   assert matches,'serialized conversation unavailable'
   for i,q in enumerate(matches):
    if q.group(1)=='user':texts.append(c[q.end():matches[i+1].start() if i+1<len(matches) else len(c)].strip())
  else:texts.append(c)
 assert texts
 return '\n\n'.join(('[USER_LATEST]' if i==len(texts)-1 else '[USER]')+'\n'+s for i,s in enumerate(texts))
def target(o):return json.loads(next(m['content'] for m in reversed(o['messages']) if m['role']=='assistant'))['user_profile']
tr=load(R/'training/data/train_sft.jsonl');va=load(R/'training/data/val_dev_sft.jsonl');bp=load(R/'training/eval_results/v2_e1_rerun2_valdev/prediction.jsonl');assert len(tr)==18657 and len(va)==len(bp)==1649
texts=[users(x) for x in tr];vt=[users(x) for x in va];ty=[target(x) for x in tr];vy=[target(x) for x in va]
vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=120000,sublinear_tf=True,dtype=np.float32);print('VECTORIZING',flush=True)
try:X=vec.fit_transform(texts);V=vec.transform(vt)
except MemoryError:
 vec.set_params(max_features=80000);X=vec.fit_transform(texts);V=vec.transform(vt)
print('FEATURES',X.shape,flush=True)
results={};models={};hyb={};allpred={}
for dim,allowed in PROFILE_LABELS.items():
 def canon(a):return tuple(sorted(set(a)))
 gt=[canon(x.get(dim,[])) for x in vy];base=[canon(x['user_profile'].get(dim,[])) for x in bp];ts=[canon(x.get(dim,[])) for x in ty];assert all(set(x)<=allowed for x in ts+gt)
 cnt=collections.Counter(ts);classes=sorted(cnt);mapping={c:i for i,c in enumerate(classes)};svc=LinearSVC(class_weight=None,random_state=50);svc.fit(X,np.array([mapping[x] for x in ts]));sp=[classes[i] for i in svc.predict(V)];sa=np.mean([x==y for x,y in zip(gt,sp)])
 labs=sorted(allowed);log=[];prob=[]
 for l in labs:
  y=np.array([int(l in x) for x in ts]);lr=LogisticRegression(class_weight='balanced',max_iter=1000,random_state=50,solver='liblinear');lr.fit(X,y);log.append(lr);prob.append(lr.predict_proba(V)[:,1])
 prob=np.stack(prob,axis=1);gr=[]
 for t in [.4,.5,.6,.7,.8]:
  pp=[tuple(l for l,q in zip(labs,row) if q>=t) for row in prob];acc=float(np.mean([x==y for x,y in zip(gt,pp)]));gr.append((acc,t,pp))
 ma,th,mp=max(gr,key=lambda x:x[0]);ba=float(np.mean([x==y for x,y in zip(gt,base)]));best='EXACT_SET' if sa>=ma else 'MULTILABEL';pred=sp if sa>=ma else mp;replace=max(sa,ma)>=ba+.01;hp=pred if replace else base
 diag={}
 for l in labs:diag[l]={'gt_positive':sum(l in x for x in gt),'v2_false_negative':sum(l in x and l not in y for x,y in zip(gt,base)),'classifier_false_negative':sum(l in x and l not in y for x,y in zip(gt,pred))}
 result={'unique_set_classes':len(cnt),'rare_set_classes':sum(n<5 for n in cnt.values()),'top_sets':[{'set':list(s) or ['__EMPTY__'],'count':n} for s,n in cnt.most_common(10)],'v2':ba,'exact_set_classifier':float(sa),'multilabel_classifier':ma,'threshold':th,'grid':[(acc,t) for acc,t,_ in gr],'winner':best if replace else 'V2_E1','hybrid_accuracy':float(np.mean([x==y for x,y in zip(gt,hp)])),'wrong_to_right':sum(g!=b and g==h for g,b,h in zip(gt,base,hp)),'right_to_wrong':sum(g==b and g!=h for g,b,h in zip(gt,base,hp)),'labels':diag};results[dim]=result;models[dim]={'svc':svc,'classes':classes,'lr':log,'labels':labs,'threshold':th,'winner':result['winner']};hyb[dim]=hp;print('DIM',dim,result['v2'],sa,ma,result['winner'],flush=True)
mean=float(np.mean([x['hybrid_accuracy'] for x in results.values()]));baseline=float(np.mean([x['v2'] for x in results.values()]));status='STRONG_PROMOTE' if mean>=.662 else 'PROMOTE' if mean>=.652 else 'PROMOTE_WEAK' if mean>=.647 else 'REJECT'
lat=[];needed=[d for d in models if models[d]['winner']!='V2_E1']
for text in vt[:100]:
 st=time.perf_counter()
 if needed:
  z=vec.transform([text])
  for dim in needed:
   d=models[dim]
   if d['winner']=='EXACT_SET':d['svc'].predict(z)
   else:[m.predict_proba(z) for m in d['lr']]
 lat.append((time.perf_counter()-st)*1000)
final={'dimensions':results,'hybrid_profile_mean':mean,'profile_mean_delta':mean-baseline,'latency_avg_ms':float(np.mean(lat)),'latency_p50_ms':float(np.percentile(lat,50)),'latency_p95_ms':float(np.percentile(lat,95)),'result':status,'emotion_changed':False,'response_changed':False,'memory_refs_changed':False,'samples':1649}
if status!='REJECT':
 out=R/'training/outputs/v2-p1-profile-classifier';out.mkdir(exist_ok=True);joblib.dump(vec,out/'vectorizer.joblib');joblib.dump(models,out/'classifiers.joblib');(out/'config.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))
(A/'result.json').write_text(json.dumps(final,ensure_ascii=False,indent=2));print('DONE',json.dumps(final,ensure_ascii=False),flush=True)
