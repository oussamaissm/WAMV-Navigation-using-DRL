import copy, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN=-20
LOG_STD_MAX=2

class MLP(nn.Module):
    def __init__(self,n_in,n_out,h=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_in,h),nn.ReLU(),
                               nn.Linear(h,h),nn.ReLU(),nn.Linear(h,n_out))
    def forward(self,x): return self.net(x)

class Actor(nn.Module):
    """Gaussian policy; tanh makes actions lie in [-1,1]."""
    def __init__(self,state_dim,action_dim,h=256):
        super().__init__(); self.net=MLP(state_dim,2*action_dim,h)
        self.action_dim=action_dim

    def forward(self,s):
        mean,log_std=torch.chunk(self.net(s),2,dim=-1)
        return mean,torch.clamp(log_std,LOG_STD_MIN,LOG_STD_MAX)

    def sample(self,s):
        mean,log_std=self(s); std=log_std.exp()
        dist=Normal(mean,std)
        z=dist.rsample()                 # reparameterization trick
        a=torch.tanh(z)                  # normalized action
        logp=dist.log_prob(z)-torch.log(1-a.pow(2)+1e-6)
        logp=logp.sum(-1,keepdim=True)
        return a,logp,torch.tanh(mean)

class Critic(nn.Module):
    """Q(s,a). Two independent critics are used by SAC."""
    def __init__(self,state_dim,action_dim,h=256):
        super().__init__(); self.q=MLP(state_dim+action_dim,1,h)
    def forward(self,s,a): return self.q(torch.cat([s,a],-1))

class SACAgent:
    def __init__(self,state_dim,action_dim,gamma=.99,tau=.005,
                 actor_lr=3e-4,critic_lr=3e-4,alpha_lr=3e-4,
                 hidden_dim=256,target_entropy=None,device=None):
        self.gamma=gamma; self.tau=tau
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor=Actor(state_dim,action_dim,hidden_dim).to(self.device)
        self.c1=Critic(state_dim,action_dim,hidden_dim).to(self.device)
        self.c2=Critic(state_dim,action_dim,hidden_dim).to(self.device)
        self.t1=copy.deepcopy(self.c1).to(self.device)
        self.t2=copy.deepcopy(self.c2).to(self.device)
        for p in self.t1.parameters(): p.requires_grad=False
        for p in self.t2.parameters(): p.requires_grad=False
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.c1o=torch.optim.Adam(self.c1.parameters(),lr=critic_lr)
        self.c2o=torch.optim.Adam(self.c2.parameters(),lr=critic_lr)
        self.target_entropy=-float(action_dim) if target_entropy is None else target_entropy
        self.log_alpha=torch.zeros(1,requires_grad=True,device=self.device)
        self.alpha_o=torch.optim.Adam([self.log_alpha],lr=alpha_lr)

    @property
    def alpha(self): return self.log_alpha.exp()

    def select_action(self,state,deterministic=False):
        s=torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad():
            a,_,mean_a=self.actor.sample(s)
        return (mean_a if deterministic else a).cpu().numpy()[0].astype(np.float32)

    def update(self,replay_buffer,batch_size=256):
        s,a,r,ns,d=replay_buffer.sample(batch_size)
        s=torch.as_tensor(s,dtype=torch.float32,device=self.device)
        a=torch.as_tensor(a,dtype=torch.float32,device=self.device)
        r=torch.as_tensor(r,dtype=torch.float32,device=self.device)
        ns=torch.as_tensor(ns,dtype=torch.float32,device=self.device)
        d=torch.as_tensor(d,dtype=torch.float32,device=self.device)

        with torch.no_grad():
            na,nlogp,_=self.actor.sample(ns)
            target=torch.min(self.t1(ns,na),self.t2(ns,na))-self.alpha.detach()*nlogp
            y=r+self.gamma*(1-d)*target

        q1=self.c1(s,a); q2=self.c2(s,a)
        l1=F.mse_loss(q1,y); l2=F.mse_loss(q2,y)
        self.c1o.zero_grad(); l1.backward(); self.c1o.step()
        self.c2o.zero_grad(); l2.backward(); self.c2o.step()

        na,logp,_=self.actor.sample(s)
        q=torch.min(self.c1(s,na),self.c2(s,na))
        actor_loss=(self.alpha.detach()*logp-q).mean()
        self.ao.zero_grad(); actor_loss.backward(); self.ao.step()

        alpha_loss=-(self.log_alpha*(logp.detach()+self.target_entropy)).mean()
        self.alpha_o.zero_grad(); alpha_loss.backward(); self.alpha_o.step()

        self._soft_update(self.c1,self.t1); self._soft_update(self.c2,self.t2)
        return dict(actor_loss=actor_loss.item(),critic1_loss=l1.item(),
                    critic2_loss=l2.item(),alpha_loss=alpha_loss.item(),
                    alpha=self.alpha.item())

    def _soft_update(self,src,tgt):
        for tp,sp in zip(tgt.parameters(),src.parameters()):
            tp.data.mul_(1-self.tau); tp.data.add_(self.tau*sp.data)

    def save(self,directory):
        os.makedirs(directory,exist_ok=True)
        torch.save(self.actor.state_dict(),f"{directory}/actor.pt")
        torch.save(self.c1.state_dict(),f"{directory}/critic1.pt")
        torch.save(self.c2.state_dict(),f"{directory}/critic2.pt")
        torch.save(self.t1.state_dict(),f"{directory}/target_critic1.pt")
        torch.save(self.t2.state_dict(),f"{directory}/target_critic2.pt")
        torch.save(self.log_alpha.detach().cpu(),f"{directory}/log_alpha.pt")

    def load(self,directory):
        kw=dict(map_location=self.device,weights_only=True)
        self.actor.load_state_dict(torch.load(f"{directory}/actor.pt",**kw))
        self.c1.load_state_dict(torch.load(f"{directory}/critic1.pt",**kw))
        self.c2.load_state_dict(torch.load(f"{directory}/critic2.pt",**kw))
        self.t1.load_state_dict(torch.load(f"{directory}/target_critic1.pt",**kw))
        self.t2.load_state_dict(torch.load(f"{directory}/target_critic2.pt",**kw))
        p=f"{directory}/log_alpha.pt"
        if os.path.exists(p): self.log_alpha.data.copy_(torch.load(p,map_location=self.device,weights_only=True))
