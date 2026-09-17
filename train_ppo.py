import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import os,csv,time
import numpy as np
import torch
from drl_wamv import WamvEnv
from ppo import PPOAgent, RolloutBuffer
import rclpy

STATE_DIM=7
ACTION_DIM=2
TOTAL_STEPS=200_000
N_STEPS=2048        # env steps collected per rollout, before each PPO update
BATCH_SIZE=64
N_EPOCHS=10          # passes over each rollout before it's discarded
GAMMA=0.99
LAM=0.95             # GAE lambda
SAVE_EVERY=20_000    # in environment steps

def main():
    rclpy.init()
    os.makedirs("results_ppo/models",exist_ok=True)
    env=WamvEnv(control_dt=0.2,max_episode_time=120.0,
                trajectory_path="results_ppo/wamv_trajectory.csv")
    buffer=RolloutBuffer(STATE_DIM,ACTION_DIM,N_STEPS)
    agent=PPOAgent(STATE_DIM,ACTION_DIM,gamma=GAMMA,lam=LAM,
                    device="cuda" if torch.cuda.is_available() else "cpu")

    f=open("results_ppo/ppo_training.csv","w",newline="")
    w=csv.writer(f)
    w.writerow(["global_step","episode","episode_step","episode_reward",
                "actor_loss","critic_loss","entropy","approx_kl"])

    state,_=env.reset()
    episode=1; ep_step=0; ep_reward=0.
    start=time.time(); global_step=0
    try:
        while global_step<TOTAL_STEPS:
            action,logp,value=agent.select_action(state)
            next_state,reward,terminated,truncated,info=env.step(action)
            global_step+=1; ep_step+=1; ep_reward+=float(reward)

            store_reward=float(reward)
            done_for_gae=float(terminated)

            if truncated and not terminated:
                # Time-limit cutoff, not a real terminal state: fold the
                # critic's estimate of the future into this step's reward
                # so GAE doesn't wrongly treat the episode as "solved" here.
                # (Mirrors the terminated-vs-truncated distinction you were
                # careful about in the SAC replay buffer.)
                bootstrap_v=agent.value(next_state)
                store_reward+=GAMMA*bootstrap_v
                done_for_gae=1.0  # still close this rollout segment cleanly

            buffer.add(state,action,logp,store_reward,value,done_for_gae)
            state=next_state

            if terminated or truncated:
                print(f"Episode {episode:04d} | steps={ep_step:4d} | reward={ep_reward:9.2f} | "
                      f"time={(time.time()-start)/60:.1f} min")
                state,_=env.reset(); episode+=1; ep_step=0; ep_reward=0.

            if buffer.full():
                # Bootstrap the value beyond the buffer's last transition.
                # If that transition was already terminal/truncated (handled
                # above), dones[-1]==1 zeroes this out inside compute_advantages
                # regardless of what we pass here.
                last_value=agent.value(state)
                buffer.compute_advantages(last_value,gamma=GAMMA,lam=LAM)
                stats=agent.update(buffer,batch_size=BATCH_SIZE,n_epochs=N_EPOCHS)
                w.writerow([global_step,episode,ep_step,ep_reward,
                            stats["actor_loss"],stats["critic_loss"],
                            stats["entropy"],stats["approx_kl"]])
                f.flush()
                print(f"Update @ step {global_step} | actor_loss={stats['actor_loss']:.4f} | "
                      f"critic_loss={stats['critic_loss']:.4f} | "
                      f"entropy={stats['entropy']:.4f} | approx_kl={stats['approx_kl']:.5f}")
                buffer.reset()

            if global_step%SAVE_EVERY==0:
                path=f"results_ppo/models/step_{global_step}"
                agent.save(path)
                print("Saved:",path)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        agent.save("results_ppo/models/final")
        f.close()
        env.close()
        print("Final model: results_ppo/models/final")

if __name__=="__main__":
    main()