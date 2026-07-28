"""
01_hello_mdp.py  —  Lesson 1: See the RL loop (MDP) LIVE.

Goal: Demonstrate the agent<->environment loop with MuJoCo's READY-MADE 'Humanoid' robot.
No TRAINING yet. We just generate RANDOM actions and print the 3 parts of the loop:
  - OBSERVATION (state): the numbers the agent 'senses'
  - ACTION: the joints it controls
  - REWARD: the score of how well it is doing
Random action = 'drunk baby' => it will fall immediately. Training's job is to turn this
randomness into a policy that 'increases the reward'.
"""
import gymnasium as gym

# 1) Create the ENVIRONMENT. This line opens a ready-made humanoid robot running in the MuJoCo physics engine.
env = gym.make("Humanoid-v5")

# 2) Initial state (reset). obs = the first observation vector.
obs, info = env.reset(seed=0)

print("=== ENVIRONMENT: Humanoid-v5 (MuJoCo's ready-made humanoid) ===")
print(f"OBSERVATION size : {env.observation_space.shape}  <- how many numbers the agent sees")
print(f"ACTION size      : {env.action_space.shape}  <- how many joints/actuators it controls")
print(f"Action range     : per joint [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
print()

# 3) LOOP: apply a random action for 12 steps, watch the loop's output.
print("=== LOOP: 12 steps, RANDOM action (NO training -> will fall) ===")
total = 0.0
for step in range(12):
    action = env.action_space.sample()                 # random control command
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    durum = "FELL (done)" if terminated else "standing"
    print(f"  step {step+1:2d}: reward={reward:+6.2f} | total={total:+8.2f} | {durum}")
    if terminated or truncated:
        print("  --> fell, environment resetting (reset)")
        obs, info = env.reset()

env.close()
print()
print("LESSON: A random action means falling every time.")
print("      In the next step we will set up OUR OWN robot and start turning this")
print("      randomness into a 'stand upright first' policy using RL.")
