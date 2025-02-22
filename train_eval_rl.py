import gymnasium as gym
import mani_skill.envs

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import torch
import imageio
import datetime

total_timesteps = 150000
SEED = 42
model_file = "models/" + "ppo_pick_cube"

np.random.seed(SEED)
# Wrap the environment to be compatible with SB3
vec_env = DummyVecEnv([lambda: gym.make('PickCube-v1', 
                                        obs_mode='state', 
                                        control_mode='pd_joint_delta_pos', 
                                        render_mode="rgb_array",
                                        )])

# Define the model (policy network)
model = PPO("MlpPolicy", vec_env, verbose=1, seed=SEED)

# Train the agent
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save(model_file)

# Load trained model
model = PPO.load(model_file)

vec_env.close()
# Create a single environment for evaluation
eval_env = gym.make("PickCube-v1", 
                    obs_mode="state", 
                    control_mode="pd_ee_delta_pos",
                    render_mode="rgb_array",
                    viewer_camera_configs=dict(shader_pack="rt-fast"),
                    # render_mode="human",
                    )

# Reset environment
obs, _ = eval_env.reset(seed=42)
done = False
frames = []

i = 0
while not done and i<1200:
    i += 1
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = eval_env.step(action)

    # capture frame
    frame = eval_env.render()
    # Fix frame shape: remove batch dimension if present
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()  # Convert Torch tensor to NumPy array
    if len(frame.shape) == 4:  # If batch dimension exists
        frame = frame.squeeze(0)  # Remove batch dim: (1, H, W, 3) → (H, W, 3)
    frames.append(frame)

eval_env.close()

# Save video
print("Saving video...")
video_path = "videos/" + "evaluation_video_" + str(datetime.datetime.now()) + ".mp4"
imageio.mimsave(video_path, frames, fps=60)

print(f"Video saved at: {video_path}")