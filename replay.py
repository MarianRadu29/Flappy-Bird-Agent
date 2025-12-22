import gymnasium as gym
import torch
import flappy_bird_gymnasium
from config import MODEL_PATH, SHOW_GAME_WINDOW
from wrapper import Wrapper
from dqn import DQN



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
render_mode = "human" if SHOW_GAME_WINDOW else "rgb_array"
env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
env = Wrapper(env)

n_actions = env.action_space.n
state, _ = env.reset()
n_channels = state.shape[2]

policy_net = DQN(n_channels, n_actions).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
policy_net.load_state_dict(checkpoint['model_state_dict'])
policy_net.eval()

state = torch.tensor(state, device=device).permute(2,0,1).unsqueeze(0)

while True:
    with torch.no_grad():
        action = policy_net(state).max(1).indices.view(1,1)

    obs, reward, terminated, truncated, info = env.step(action.item())

    if terminated or truncated:
        state, _ = env.reset()
        state = torch.tensor(state, device=device).permute(2,0,1).unsqueeze(0)
    else:
        state = torch.tensor(obs, device=device).permute(2,0,1).unsqueeze(0)
