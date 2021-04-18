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

    def sample_action(self, obs, epsilon):
        action_probabilities = self.sm(self.network.forward(torch.FloatTensor([obs])))
        max_action_probability, max_action_index = torch.max(action_probabilities, dim=1)

        if torch.rand(1,).item() > epsilon:
            return max_action_index.item()
        else:
            action_probabilities = action_probabilities.data.numpy()[0]
            return np.random.choice(len(action_probabilities),p=action_probabilities)
    
    def choose_action(self, obs):
        with torch.no_grad():
            action_probabilities = self.sm(self.network.forward(torch.FloatTensor([obs])))
            max_action_probability, max_action_index = torch.max(action_probabilities, dim=1)

            return max_action_index.item()

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
    
    def load_pretrained_model(self, model_path="model/grab-coin-cross_entropy.pth"):
        self.network.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="model/grab-coin-cross_entropy.pth"):
        torch.save(self.network.state_dict(), model_path)