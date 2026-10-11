"""Rank-local deterministic 3 response / 1 emotion microbatch stream."""
import random
from torch.utils.data import IterableDataset
class DualObjectiveStream(IterableDataset):
 def __init__(self,response,emotion,rank,world_size,steps=160,grad_acc=8,seed=61):
  self.response=response;self.emotion=emotion;self.rank=rank;self.world_size=world_size;self.length=steps*grad_acc;self.seed=seed
 def __len__(self):return self.length
 def schedule(self):
  rng=random.Random(self.seed);r=list(range(len(self.response)));e=list(range(len(self.emotion)));rng.shuffle(r);rng.shuffle(e);rp=ep=0
  for batch in range(self.length):
   objective='emotion' if batch%4==3 else 'response';pool=e if objective=='emotion' else r;pos=ep if objective=='emotion' else rp
   index=pool[(pos*self.world_size+self.rank)%len(pool)]
   if objective=='emotion':ep+=1
   else:rp+=1
   yield objective,index
 def __iter__(self):
  for objective,index in self.schedule():yield (self.emotion if objective=='emotion' else self.response)[index]
