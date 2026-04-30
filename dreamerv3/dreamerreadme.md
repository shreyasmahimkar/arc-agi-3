# DreamerV3 + Neural Map Implementation Walkthrough

I have successfully implemented the "glass box" DreamerV3 agent that uses Neural Maps to solve the `ls20` environment's partial observability challenges. 

The entire framework was built completely from scratch and optimized to run on your Mac (using the `mps` GPU backend natively). 

## 1. The Environment Adapter (Gym Wrapper)
`ls20_dreamer_env.py`

We built a custom Gymnasium wrapper that seamlessly hooks into the ARC-AGI-3 `ls20` backend.

* **The Neural Map**: At every step, the environment looks at the `visible_frame` (where unseen areas are black/0) and extracts all non-zero pixels. It stamps these pixels onto its persistent 64x64 internal tensor memory. This guarantees the agent never forgets what it saw in a corner.
* **Reward Shaping**: We meticulously shaped the reward so the agent learns to preserve fuel:
  * `-0.01` penalty per step (to encourage speed).
  * `-10.0` penalty when fuel runs out and a life is lost.
  * `+10.0` reward upon clearing a level.

## 2. The World Model (RSSM)
`models/dreamer/rssm.py`

We built a lightweight version of the Dreamer Recurrent State-Space Model.

* **The Encoder**: A CNN that takes in a 2-channel 64x64 matrix (the raw frame + the Neural Map) so it understands both its immediate surroundings and its long-term memory.
* **The Latent Dynamics**: It uses a GRU and a Gaussian prior/posterior model to learn the "physics" of `ls20`.
* **Predictors**: It trains itself to predict rewards (`reward_net`) and termination (`continue_net`) purely from the latent state.

## 3. The Actor-Critic
`models/dreamer/actor_critic.py`

We built a twin-MLP architecture. The beauty of Dreamer is that this Actor-Critic network never trains on real environment data. It learns entirely from the RSSM's "imagined" rollouts.

## 4. The Online Orchestrator
`play_and_learn.py`

This is the main script. It proves the concept of Online Learning. It drops the agent into the game. The agent explores, adds experiences to a Replay Buffer, and simultaneously updates its World Model and Policy on-the-fly.

> [!TIP]
> **Running the Agent**
> I've verified this script runs perfectly on Mac (it automatically detects the Apple Silicon `mps` accelerator).
> 
> You can run the agent yourself:
> ```bash
> source .venv312/bin/activate
> python play_and_learn.py
> ```
> Add the `--render_memory` flag to open a real-time matplotlib window that shows the Neural Map filling up as the agent explores the dark!
