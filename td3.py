import copy, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,n_in,n_out,h=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_in,h),nn.ReLU(),
                               nn.Linear(h,h),nn.ReLU(),nn.Linear(h,n_out))
    def forward(self,x): return self.net(x)

class Actor(nn.Module):
    """Deterministic policy; tanh bounds the action to [-1,1]. Unlike SAC's
    Actor there is no distribution here — select_action() is what adds
    exploration noise, externally, at rollout time."""
    def __init__(self,state_dim,action_dim,h=256):
        super().__init__(); self.net=MLP(state_dim,action_dim,h)
    def forward(self,s): return torch.tanh(self.net(s))

class Critic(nn.Module):
    """Q(s,a). Same shape as SAC's critic: two independent instances of
    this are used for clipped double-Q learning."""
    def __init__(self,state_dim,action_dim,h=256):
        super().__init__(); self.q=MLP(state_dim+action_dim,1,h)
    def forward(self,s,a): return self.q(torch.cat([s,a],-1))

class TD3Agent:
    def __init__(self,state_dim,action_dim,gamma=.99,tau=.005,
                 actor_lr=3e-4,critic_lr=3e-4,hidden_dim=256,
                 policy_noise=0.2,noise_clip=0.5,policy_freq=2,
                 exploration_noise=0.1,device=None):
        self.gamma=gamma; self.tau=tau
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.actor=Actor(state_dim,action_dim,hidden_dim).to(self.device)
        self.actor_target=copy.deepcopy(self.actor).to(self.device)
        self.c1=Critic(state_dim,action_dim,hidden_dim).to(self.device)
        self.c2=Critic(state_dim,action_dim,hidden_dim).to(self.device)
        self.t1=copy.deepcopy(self.c1).to(self.device)
        self.t2=copy.deepcopy(self.c2).to(self.device)
        for p in self.actor_target.parameters(): p.requires_grad=False
        for p in self.t1.parameters(): p.requires_grad=False
        for p in self.t2.parameters(): p.requires_grad=False

        self.ao=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.c1o=torch.optim.Adam(self.c1.parameters(),lr=critic_lr)
        self.c2o=torch.optim.Adam(self.c2.parameters(),lr=critic_lr)

        # policy_noise/noise_clip: target-policy-smoothing noise added to the
        #   *target* actor's action when building the critic's TD target.
        # exploration_noise: separate, larger noise added to the *live*
        #   actor's action during data collection (this is TD3's stand-in
        #   for SAC's stochastic policy).
        self.policy_noise=policy_noise; self.noise_clip=noise_clip
        self.policy_freq=policy_freq; self.exploration_noise=exploration_noise
        self.action_dim=action_dim
        self.total_updates=0

    def select_action(self,state,noise=True):
        s=torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad():
            a=self.actor(s).cpu().numpy()[0]
        if noise:
            a=a+np.random.normal(0,self.exploration_noise,size=a.shape)
        return np.clip(a,-1.0,1.0).astype(np.float32)

    def update(self,replay_buffer,batch_size=256):
        self.total_updates+=1
        s,a,r,ns,d=replay_buffer.sample(batch_size)
        s=torch.as_tensor(s,dtype=torch.float32,device=self.device)
        a=torch.as_tensor(a,dtype=torch.float32,device=self.device)
        r=torch.as_tensor(r,dtype=torch.float32,device=self.device)
        ns=torch.as_tensor(ns,dtype=torch.float32,device=self.device)
        d=torch.as_tensor(d,dtype=torch.float32,device=self.device)

        with torch.no_grad():
            # Target policy smoothing: perturb the target action so the
            # critic can't be exploited by narrow, spurious Q-value peaks.
            noise=(torch.randn_like(a)*self.policy_noise).clamp(-self.noise_clip,self.noise_clip)
            na=(self.actor_target(ns)+noise).clamp(-1.0,1.0)
            # Clipped double-Q: take the min of the two target critics to
            # curb the overestimation bias that plain actor-critic suffers from.
            target_q=torch.min(self.t1(ns,na),self.t2(ns,na))
            y=r+self.gamma*(1-d)*target_q

        q1=self.c1(s,a); q2=self.c2(s,a)
        l1=F.mse_loss(q1,y); l2=F.mse_loss(q2,y)
        self.c1o.zero_grad(); l1.backward(); self.c1o.step()
        self.c2o.zero_grad(); l2.backward(); self.c2o.step()

        actor_loss=None
        # Delayed policy updates: only touch the actor and the target
        # networks every policy_freq critic updates, so the critic has a
        # chance to settle before the actor starts chasing it.
        if self.total_updates%self.policy_freq==0:
            actor_loss=-self.c1(s,self.actor(s)).mean()
            self.ao.zero_grad(); actor_loss.backward(); self.ao.step()

            self._soft_update(self.actor,self.actor_target)
            self._soft_update(self.c1,self.t1)
            self._soft_update(self.c2,self.t2)
            actor_loss=actor_loss.item()

        return dict(actor_loss=actor_loss,critic1_loss=l1.item(),critic2_loss=l2.item())

    def _soft_update(self,src,tgt):
        for tp,sp in zip(tgt.parameters(),src.parameters()):
            tp.data.mul_(1-self.tau); tp.data.add_(self.tau*sp.data)

    def save(self,directory):
        os.makedirs(directory,exist_ok=True)
        torch.save(self.actor.state_dict(),f"{directory}/actor.pt")
        torch.save(self.actor_target.state_dict(),f"{directory}/actor_target.pt")
        torch.save(self.c1.state_dict(),f"{directory}/critic1.pt")
        torch.save(self.c2.state_dict(),f"{directory}/critic2.pt")
        torch.save(self.t1.state_dict(),f"{directory}/target_critic1.pt")
        torch.save(self.t2.state_dict(),f"{directory}/target_critic2.pt")

    def load(self,directory):
        kw=dict(map_location=self.device,weights_only=True)
        self.actor.load_state_dict(torch.load(f"{directory}/actor.pt",**kw))
        self.actor_target.load_state_dict(torch.load(f"{directory}/actor_target.pt",**kw))
        self.c1.load_state_dict(torch.load(f"{directory}/critic1.pt",**kw))
        self.c2.load_state_dict(torch.load(f"{directory}/critic2.pt",**kw))
        self.t1.load_state_dict(torch.load(f"{directory}/target_critic1.pt",**kw))
        self.t2.load_state_dict(torch.load(f"{directory}/target_critic2.pt",**kw))
