import os,torch
r=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(r);x=torch.ones(4,4,device='cuda:'+str(r));x@x;torch.cuda.synchronize(r);print('RANK_PASS',r,flush=True)
