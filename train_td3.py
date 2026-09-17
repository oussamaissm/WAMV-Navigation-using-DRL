import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import os,csv,time
import numpy as np
import torch
from drl_wamv import WamvEnv
from reply_buffer import ReplayBuffer
from td3 import TD3Agent
import rclpy

STATE_DIM=7
ACTION_DIM=2
TOTAL_STEPS=100_000
WARMUP_STEPS=1000
BATCH_SIZE=256
BUFFER_SIZE=100_000
SAVE_EVERY=5_000

def main():
    rclpy.init()
    os.makedirs("results_td3/models",exist_ok=True)
    env=WamvEnv(control_dt=0.2,max_episode_time=120.0,
                trajectory_path="results_td3/wamv_trajectory.csv")
    buffer=ReplayBuffer(STATE_DIM,ACTION_DIM,BUFFER_SIZE)
    agent=TD3Agent(STATE_DIM,ACTION_DIM,device="cuda" if torch.cuda.is_available() else "cpu")
    f=open("results_td3/td3_training.csv","w",newline="")
    w=csv.writer(f)
    w.writerow(["global_step","episode","episode_step","episode_reward","reward",
                "buffer_size","actor_loss","critic1_loss","critic2_loss"])
    state,_=env.reset()
    episode=1; ep_step=0; ep_reward=0.; last_actor_loss=None
    start=time.time()
    try:
        for step in range(1,TOTAL_STEPS+1):
            if step<=WARMUP_STEPS:
                action=np.random.uniform(-1,1,ACTION_DIM).astype(np.float32)
            else:
                action=agent.select_action(state)  # noise=True by default

            next_state,reward,terminated,truncated,info=env.step(action)
            # As with SAC, only a physical terminal state is stored as done;
            # a time-limit truncation is not treated as terminal.
            buffer.add(state,action,reward,next_state,float(terminated))
            state=next_state; ep_step+=1; ep_reward+=float(reward)

            if step>WARMUP_STEPS and len(buffer)>=BATCH_SIZE:
                stats=agent.update(buffer,BATCH_SIZE)
                # actor_loss is only produced every policy_freq critic
                # updates (delayed policy update) — carry the last known
                # value forward on the steps in between, for logging.
                if stats["actor_loss"] is not None:
                    last_actor_loss=stats["actor_loss"]
                w.writerow([step,episode,ep_step,ep_reward,float(reward),len(buffer),
                            last_actor_loss,stats["critic1_loss"],stats["critic2_loss"]])

            if terminated or truncated:
                print(f"Episode {episode:04d} | steps={ep_step:4d} | reward={ep_reward:9.2f} | "
                      f"buffer={len(buffer):6d} | "
                      f"time={(time.time()-start)/60:.1f} min")
                state,_=env.reset(); episode+=1; ep_step=0; ep_reward=0.

            if step%SAVE_EVERY==0:
                path=f"results_td3/models/step_{step}"
                agent.save(path); f.flush()
                print("Saved:",path)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        agent.save("results_td3/models/final")
        buffer.save("results_td3/replay_buffer.npz")
        f.close()
        env.close()
        print("Final model: results_td3/models/final")

if __name__=="__main__":
    main()