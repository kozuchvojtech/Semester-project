import torch as torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from network import Network

class Agent():
    def __init__(self, layer_sizes, seed, learning_rate):
        torch.manual_seed(seed)

        self.network = Network(layer_sizes)
        self.objective_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=learning_rate)
        self.sm = nn.Softmax(dim=1)

    def sample_action(self, obs):
        obs_v = torch.FloatTensor([obs])

        act_prob_v = self.sm(self.network(obs_v))
        act_prob = act_prob_v.data.numpy()[0]

        return np.random.choice(len(act_prob),p=act_prob)

    def train(self, elite_batch, obs, act):
        obs_v = torch.FloatTensor(obs)
        act_v = torch.LongTensor(act)
        elite_batch = elite_batch[-500:]

        self.optimizer.zero_grad()
        action_scores_v = self.network(obs_v)
        loss_v = self.objective_func(action_scores_v,act_v)
        loss_v.backward()
        self.optimizer.step()

        return loss_v.item()