import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN=-20
LOG_STD_MAX=2

class MLP(nn.Module):
    def __init__(self,n_in,n_out,h=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_in,h),nn.Tanh(),
                               nn.Linear(h,h),nn.Tanh(),nn.Linear(h,n_out))
    def forward(self,x): return self.net(x)

class Actor(nn.Module):
    """Gaussian policy with a *state-independent* log_std (standard for PPO
    continuous control). Unlike SAC's Actor, there is no tanh squashing on
    the action itself: the env already clips to [-1,1], and an unbounded
    Gaussian keeps log_prob/entropy exact and simple, which matters because
    PPO's ratio term needs old and new log-probs to be computed consistently."""
    def __init__(self,state_dim,action_dim,h=256):
        super().__init__()
        self.mean_net=MLP(state_dim,action_dim,h)
        self.log_std=nn.Parameter(torch.zeros(action_dim)-0.5)  # std starts ~0.6

    def forward(self,s):
        mean=self.mean_net(s)
        log_std=self.log_std.clamp(LOG_STD_MIN,LOG_STD_MAX)
        return mean,log_std.exp()

    def sample(self,s):
        mean,std=self(s)
        dist=Normal(mean,std)
        a=dist.rsample()
        logp=dist.log_prob(a).sum(-1)
        return a,logp

    def log_prob_entropy(self,s,a):
        mean,std=self(s)
        dist=Normal(mean,std)
        logp=dist.log_prob(a).sum(-1)
        entropy=dist.entropy().sum(-1)
        return logp,entropy

class Critic(nn.Module):
    """V(s). PPO uses a state-value function, not Q(s,a): there's no
    target network and no second critic, since there's no TD target
    computed from a bootstrapped action sample."""
    def __init__(self,state_dim,h=256):
        super().__init__(); self.v=MLP(state_dim,1,h)
    def forward(self,s): return self.v(s).squeeze(-1)


class RolloutBuffer:
    """Fixed-size on-policy buffer holding exactly one rollout of n_steps.
    Filled once, used for a few epochs of updates, then reset — unlike
    SACAgent's ReplayBuffer, nothing here is ever reused across rollouts."""
    def __init__(self,state_dim,action_dim,n_steps):
        self.n_steps=n_steps
        self.states=np.zeros((n_steps,state_dim),dtype=np.float32)
        self.actions=np.zeros((n_steps,action_dim),dtype=np.float32)
        self.logprobs=np.zeros(n_steps,dtype=np.float32)
        self.rewards=np.zeros(n_steps,dtype=np.float32)
        self.values=np.zeros(n_steps,dtype=np.float32)
        self.dones=np.zeros(n_steps,dtype=np.float32)  # true episode end for GAE
        self.ptr=0

    def add(self,s,a,logp,r,v,done):
        i=self.ptr
        self.states[i]=s; self.actions[i]=a; self.logprobs[i]=logp
        self.rewards[i]=r; self.values[i]=v; self.dones[i]=done
        self.ptr+=1

    def full(self): return self.ptr>=self.n_steps

    def compute_advantages(self,last_value,gamma=0.99,lam=0.95):
        """Generalized Advantage Estimation, computed backward through the
        rollout. `dones[t]==1` zeroes the bootstrap across that boundary —
        this must only be set for a genuine terminal state. Time-limit
        truncations are handled upstream (see train_ppo.py) by folding the
        bootstrap value directly into that step's stored reward and then
        marking it done here too, so the backward pass still closes cleanly."""
        adv=np.zeros(self.n_steps,dtype=np.float32)
        last_gae=0.0
        for t in reversed(range(self.n_steps)):
            next_value=last_value if t==self.n_steps-1 else self.values[t+1]
            next_nonterminal=1.0-self.dones[t]
            delta=self.rewards[t]+gamma*next_value*next_nonterminal-self.values[t]
            last_gae=delta+gamma*lam*next_nonterminal*last_gae
            adv[t]=last_gae
        self.advantages=adv
        self.returns=adv+self.values

    def get_batches(self,batch_size,device):
        idx=np.arange(self.n_steps); np.random.shuffle(idx)
        adv=(self.advantages-self.advantages.mean())/(self.advantages.std()+1e-8)
        for start in range(0,self.n_steps,batch_size):
            b=idx[start:start+batch_size]
            yield (torch.as_tensor(self.states[b],dtype=torch.float32,device=device),
                   torch.as_tensor(self.actions[b],dtype=torch.float32,device=device),
                   torch.as_tensor(self.logprobs[b],dtype=torch.float32,device=device),
                   torch.as_tensor(adv[b],dtype=torch.float32,device=device),
                   torch.as_tensor(self.returns[b],dtype=torch.float32,device=device))

    def reset(self): self.ptr=0


class PPOAgent:
    def __init__(self,state_dim,action_dim,gamma=.99,lam=.95,clip_eps=.2,
                 actor_lr=3e-4,critic_lr=1e-3,entropy_coef=0.0,value_coef=.5,
                 max_grad_norm=.5,hidden_dim=256,device=None):
        self.gamma=gamma; self.lam=lam; self.clip_eps=clip_eps
        self.entropy_coef=entropy_coef; self.value_coef=value_coef
        self.max_grad_norm=max_grad_norm
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor=Actor(state_dim,action_dim,hidden_dim).to(self.device)
        self.critic=Critic(state_dim,hidden_dim).to(self.device)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.co=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)

    def select_action(self,state):
        """Returns (action, logprob, value) — PPO needs all three stored
        per-step, since the update later needs the *old* logprob for the
        ratio, and the *old* value for the advantage/return targets."""
        s=torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad():
            a,logp=self.actor.sample(s)
            v=self.critic(s)
        return a.cpu().numpy()[0].astype(np.float32),float(logp.item()),float(v.item())

    def value(self,state):
        s=torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.critic(s).item())

    def update(self,buffer,batch_size=64,n_epochs=10):
        last=None
        for _ in range(n_epochs):
            for s,a,old_logp,adv,ret in buffer.get_batches(batch_size,self.device):
                logp,entropy=self.actor.log_prob_entropy(s,a)
                ratio=torch.exp(logp-old_logp)
                surr1=ratio*adv
                surr2=torch.clamp(ratio,1-self.clip_eps,1+self.clip_eps)*adv
                actor_loss=-torch.min(surr1,surr2).mean()-self.entropy_coef*entropy.mean()

                value=self.critic(s)
                critic_loss=((value-ret)**2).mean()

                self.ao.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
                self.ao.step()

                self.co.zero_grad(); (self.value_coef*critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
                self.co.step()

                last=dict(actor_loss=actor_loss.item(),critic_loss=critic_loss.item(),
                           entropy=entropy.mean().item(),
                           approx_kl=(old_logp-logp).mean().item())
        return last

    def save(self,directory):
        os.makedirs(directory,exist_ok=True)
        torch.save(self.actor.state_dict(),f"{directory}/actor.pt")
        torch.save(self.critic.state_dict(),f"{directory}/critic.pt")

    def load(self,directory):
        kw=dict(map_location=self.device,weights_only=True)
        self.actor.load_state_dict(torch.load(f"{directory}/actor.pt",**kw))
        self.critic.load_state_dict(torch.load(f"{directory}/critic.pt",**kw))
