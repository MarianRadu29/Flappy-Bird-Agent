import gymnasium as gym
import torch
import flappy_bird_gymnasium
import os

from dqn import DQN
from config import MODEL_PATH
from wrapper import Wrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make(
        "FlappyBird-v0",
        render_mode="human",
        use_lidar=False
    )
env = Wrapper(env)

n_actions = env.action_space.n

state, _ = env.reset()

n_channels = state.shape[2]

policy_net = DQN(n_channels, n_actions).to(DEVICE)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if "model_state_dict" in checkpoint:
        policy_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        policy_net.load_state_dict(checkpoint)
else:
    print("Model not found!")
    env.close()
    exit(1)

policy_net.eval()

state_tensor = torch.tensor(
        state, dtype=torch.float32, device=DEVICE
).permute(2, 0, 1).unsqueeze(0)

while True:
    # select action
    with torch.no_grad():
        action = policy_net(state_tensor).max(1).indices.item()

    next_state, reward, terminated, truncated, info = env.step(action)

    state_tensor = torch.tensor(
            next_state, dtype=torch.float32, device=DEVICE
    ).permute(2, 0, 1).unsqueeze(0)

    if terminated or truncated:
        print(f"Score: {info.get('score', 0)}")

        state, _ = env.reset()

        state_tensor = torch.tensor(
                state, dtype=torch.float32, device=DEVICE
        ).permute(2, 0, 1).unsqueeze(0)

env.close()
