from tqdm import tqdm
from time import sleep
from dqn_agent import DQN_Agent

import numpy as np

input_dim = 1
output_dim = 9
exp_replay_size = 256

agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 16, output_dim], lr=1e-3, sync_freq=5, exp_replay_size=exp_replay_size)
agent.load_pretrained_model("shortest-path-dqn.pth")

goal = 7

obs = 4
done = False
steps = [obs]

while not obs == goal:
    A = agent.get_action(np.array([obs]), 9, epsilon=0)
    print(str(obs)+' -> '+str(A.item()))

    obs = A.item()
    steps.append(A.item())

print(steps)

# for obs in range(9):
#     A = agent.get_action(np.array([obs]), 9, epsilon=0)
#     print(str(obs)+' -> '+str(A.item()))