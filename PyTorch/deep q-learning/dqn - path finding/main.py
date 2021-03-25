from dqn_agent import DQN_Agent
from tqdm import tqdm

import networkx as nx
import numpy as np
import pylab as plt
import pandas as pd

input_dim = 1
output_dim = 9
exp_replay_size = 256

agent = DQN_Agent(seed=0, layer_sizes=[input_dim, 16, output_dim], lr=1e-3, sync_freq=5, exp_replay_size=exp_replay_size)

# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
episodes = 10000
epsilon = 1

# Initialize the graph
edge_list = [(0,2),(0,1),(0,3),(2,4),(5,6),(7,4),(0,6),(5,3),(3,7),(0,8)]
goal = 7
SIZE_MATRIX = 9

G = nx.Graph()
G.add_edges_from(edge_list)

# Show the graph
position = nx.spring_layout(G)

nx.draw_networkx_nodes(G, position)
nx.draw_networkx_edges(G, position)
nx.draw_networkx_labels(G, position)
plt.show()

# Initialize R-matrix

R = np.matrix(np.ones(shape=(SIZE_MATRIX,SIZE_MATRIX)))
R *= -1

for edge in edge_list:
    if edge[1] == goal:
        #R[0,1]
        R[edge] = 100
    else:
        R[edge] = 0
    if edge[0] == goal:
        #R[1,0]
        R[edge[::-1]] = 100
    else:
        R[edge[::-1]] = 0

R[goal,goal] = 100

def get_available_actions(state):
    current_state_row = R[state,]
    available_actions = np.where(current_state_row >= 0)[1]

    return available_actions

# exploration
index = 0

for i in range(exp_replay_size):    
    current_state = np.random.randint(0, 9)

    while current_state == goal:
        current_state = np.random.randint(0, 9)

    done = False

    while not done:
        available_actions = get_available_actions(current_state)
        A = agent.get_action(np.array([current_state]), available_actions, epsilon=1)

        obs_next, reward, done = A.item(), R[current_state,A.item()], A.item() == goal or R[current_state,A.item()] < 0

        agent.collect_experience([np.array([current_state]), A.item(), reward, np.array([obs_next])])
        current_state = obs_next
        index += 1
        if index > exp_replay_size:
            break

# exploitation

for i in tqdm(range(episodes)):
    current_state = np.random.randint(0, 9)

    while current_state == goal:
        current_state = np.random.randint(0, 9)

    done, losses, ep_len, rew = False, 0, 0, 0

    while not done:
        ep_len += 1
        available_actions = get_available_actions(current_state)
        A = agent.get_action(np.array([current_state]), available_actions, epsilon)
        obs_next, reward, done = A.item(), R[current_state,A.item()], A.item() == goal or R[current_state,A.item()] < 0

        agent.collect_experience([np.array([current_state]), A.item(), reward, np.array([obs_next])])

        current_state = obs_next
        rew += reward

        loss = agent.train(batch_size=16)
        losses += loss       
        
    if (i%100 == 0):
        print('MSE: '+str(loss))

    if epsilon > 0.05:
        epsilon -= (1 / 5000)

    losses_list.append(losses / ep_len), reward_list.append(rew)
    episode_len_list.append(ep_len), epsilon_list.append(epsilon)

print("Saving trained model")
agent.save_trained_model("shortest-path-dqn.pth")