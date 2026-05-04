from __future__ import annotations
from dataclasses import dataclass
import math
from collections import Counter

@dataclass
class BraidMetrics:
    raw_length:int; reduced_length:int; writhe:int; generator_entropy:float; permutation_disorder:float

class BraidTracker:
    def __init__(self,n_strands:int):
        self.n_strands=n_strands; self.word=[]; self.reduced_word=[]; self.permutation=list(range(n_strands)); self.events=[]
    def copy(self):
        b=BraidTracker(self.n_strands); b.word=list(self.word); b.reduced_word=list(self.reduced_word); b.permutation=list(self.permutation); b.events=list(self.events); return b
    @staticmethod
    def angular_order(positions, center=None):
        if center is None: center=positions[0]
        vals=[]
        for i,p in enumerate(positions):
            vals.append((math.atan2(p[1]-center[1],p[0]-center[0]),i))
        return [i for _,i in sorted(vals)]
    @classmethod
    def generator_from_action(cls, positions, action):
        kind,i,j=action
        if kind=='noop' or i==j: return None
        order=cls.angular_order(positions); rank={node:r for r,node in enumerate(order)}
        ri,rj=rank.get(i,-1),rank.get(j,-1)
        if ri<0 or rj<0: return None
        gen=max(1,min(len(order)-1,min(ri,rj)+1)); direction=1 if (ri-rj)*(1 if kind in ('form','react') else -1)>=0 else -1
        return gen,direction
    def add_action(self, positions, action, step=0, meta=None):
        g=self.generator_from_action(positions,action)
        if not g: return None
        gen,direction=g; event=(gen,direction)
        self.word.append(event)
        if self.reduced_word and self.reduced_word[-1][0]==gen and self.reduced_word[-1][1]==-direction: self.reduced_word.pop()
        else: self.reduced_word.append(event)
        a=gen-1; b=gen
        if b < len(self.permutation): self.permutation[a],self.permutation[b]=self.permutation[b],self.permutation[a]
        ev={'step':step,'gen':gen,'dir':direction,'kind':action[0],'i':action[1],'j':action[2]}
        if meta: ev.update(meta)
        self.events.append(ev); return event
    def metrics(self):
        cnt=Counter(g for g,d in self.reduced_word); total=sum(cnt.values())
        ent=0.0
        for c in cnt.values():
            p=c/max(1,total); ent-=p*math.log(p+1e-12)
        disorder=sum(abs(i-p) for i,p in enumerate(self.permutation))/max(1,self.n_strands**2)
        return BraidMetrics(len(self.word),len(self.reduced_word),sum(d for g,d in self.reduced_word),float(ent),float(disorder))
    def word_string(self, reduced=True, max_terms=60):
        w=self.reduced_word if reduced else self.word
        out=[]
        for g,d in w[:max_terms]: out.append(f'σ{g}' + ('' if d>0 else '⁻¹'))
        return ' '.join(out) + (' ...' if len(w)>max_terms else '')

def braid_reward_correction(metrics, lambda_braid=0.005, lambda_writhe=0.001, lambda_entropy=0.0):
    return -lambda_braid*metrics.reduced_length - lambda_writhe*abs(metrics.writhe) + lambda_entropy*metrics.generator_entropy
