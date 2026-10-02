"""Train-only patch probabilities fitted to channel pixel prevalence."""
import numpy as np
from scipy.optimize import minimize
from torch.utils.data import Sampler


def fit_probabilities(positive,valid,regularization=.005,uniform_mix=.1):
    p=np.asarray(positive,np.float64);v=np.asarray(valid,np.float64)
    if p.shape!=v.shape or p.ndim!=2 or np.any(p<0) or np.any(p>v):raise ValueError('Invalid counts')
    if np.any(v.sum(0)==0):raise ValueError('No valid training pixels in a channel')
    n,c=p.shape
    # Parameters tilt whole-patch probability by the 16 coverage features.
    x=np.column_stack((np.divide(p,v,out=np.zeros_like(p),where=v>0),np.ones(n)))
    scale=v.mean(0);p=p/scale;v=v/scale
    before=p.sum(0)/v.sum(0)
    def probabilities(theta):
        score=x@theta;active=np.abs(score)<np.log(10.)
        w=np.exp(np.clip(score,-np.log(10.),np.log(10.)));soft=w/w.sum()
        return (1-uniform_mix)*soft+uniform_mix/n,soft,active
    def objective(theta):
        q,soft,active=probabilities(theta);den=q@v;r=(q@p)/den
        kl=np.sum(q*np.log(q*n))
        excess=np.maximum(np.abs(r-.5)-np.abs(before-.5),0)
        loss=np.mean((r-.5)**2)+10*np.mean(excess**2)+regularization*kl
        dr=(2*(r-.5)+20*excess*np.sign(r-.5))/c
        g=((p-v*r)@(dr/den))+regularization*(np.log(q*n)+1)
        grad=x.T@((1-uniform_mix)*soft*(g-g@soft)*active)
        return loss,grad
    result=minimize(objective,np.zeros(c+1),jac=True,method='L-BFGS-B',options=dict(maxiter=160,ftol=1e-11))
    q,_,_=probabilities(result.x);natural=np.full(n,1/n)
    # Prevent balancing from collapsing training to a tiny repeated subset.
    mix=min(1.,9/max(q.max()*n-1,1e-12),np.sqrt((4/n-1/n)/max(q@q-1/n,1e-15)))
    q=mix*q+(1-mix)*natural
    expected=(q@p)/(q@v);before=p.sum(0)/v.sum(0)
    if np.mean((expected-.5)**2)>np.mean((before-.5)**2)+1e-10:raise ValueError('Sampling worsened balance objective')
    return q,dict(method='bounded log-linear patch weighting + entropy regularization + uniform mixture',
        optimization_success=bool(result.success),optimization_message=str(result.message),iterations=int(result.nit),
        regularization=regularization,uniform_mixture=uniform_mix,extra_natural_mixture=float(1-mix),max_relative_probability_limit=10.,minimum_effective_sample_fraction=.25,no_worsening_soft_penalty=10.,effective_sample_size=float(1/(q@q)),
        min_relative_probability=float(q.min()*n),max_relative_probability=float(q.max()*n),
        natural_positive_fraction=before.tolist(),expected_positive_fraction=expected.tolist(),
        channel_max_patch_positive_fraction=np.max(x[:,:c],axis=0).tolist(),
        target_positive_fraction=.5,exact_balance_guaranteed=False)


class DistributedWeightedPatchSampler(Sampler):
    def __init__(self,probabilities,num_replicas,rank,seed=42,num_samples=None,with_draw_ids=False):
        self.probabilities=np.asarray(probabilities,np.float64)
        if self.probabilities.ndim!=1 or not np.isfinite(self.probabilities).all() or (self.probabilities<=0).any():raise ValueError('Invalid patch probabilities')
        self.probabilities/=self.probabilities.sum()
        self.world=num_replicas;self.rank=rank;self.seed=seed;self.epoch=0;self.with_draw_ids=with_draw_ids
        self.total=len(self.probabilities) if num_samples is None else num_samples
        if self.total%num_replicas or not 0<=rank<num_replicas:raise ValueError('Uneven DDP sample count')
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __len__(self):return self.total//self.world
    def global_indices(self):
        return np.random.default_rng(self.seed+self.epoch).choice(len(self.probabilities),self.total,replace=True,p=self.probabilities)
    def __iter__(self):
        indices=self.global_indices()
        if self.with_draw_ids:return iter((int(indices[j]),j) for j in range(self.rank,self.total,self.world))
        return iter(indices[self.rank::self.world].tolist())
