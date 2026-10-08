import numpy as np

class ReplayBuffer:
    """Experience replay buffer: (s, a, r, s_next, done)."""
    def __init__(self, state_dim, action_dim, capacity=100_000):
        self.capacity=int(capacity); self.ptr=0; self.size=0
        self.states=np.zeros((capacity,state_dim),np.float32)
        self.actions=np.zeros((capacity,action_dim),np.float32)
        self.rewards=np.zeros((capacity,1),np.float32)
        self.next_states=np.zeros((capacity,state_dim),np.float32)
        self.dones=np.zeros((capacity,1),np.float32)

    def add(self,state,action,reward,next_state,done):
        i=self.ptr
        self.states[i]=state; self.actions[i]=action
        self.rewards[i]=reward; self.next_states[i]=next_state
        self.dones[i]=done
        self.ptr=(i+1)%self.capacity
        self.size=min(self.size+1,self.capacity)

    def sample(self,batch_size):
        if self.size<batch_size: raise ValueError("Replay buffer too small")
        i=np.random.randint(0,self.size,batch_size)
        return self.states[i],self.actions[i],self.rewards[i],self.next_states[i],self.dones[i]

    def __len__(self): return self.size

    def save(self,path):
        np.savez_compressed(path,states=self.states[:self.size],
            actions=self.actions[:self.size],rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],dones=self.dones[:self.size])

