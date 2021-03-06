import torch as torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from network import Network

class Agent():
    def __init__(self, layer_sizes, seed, learning_rate):
        self.network = Network(layer_sizes)
        self.objective_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=learning_rate)
        self.sm = nn.Softmax(dim=1)

    def sample_action(self, obs, epsilon):
        """Method for next actions resolving based on the current observation received. Based on epsilon value the agent either rather randomly selects an action with probability distribution or selects the action which is the most probable one.

        Args:
            obs (array): current observation
            epsilon (float): a scalar used for epsilon-greedy

        Returns:
            integer: the action taken by the agent in the next step
        """
        actions = self.network(torch.FloatTensor([obs]))
        actions = self.sm(actions)

        if torch.rand(1,).item() > epsilon:
            return torch.argmax(actions, dim=1).item()
        else:
            action_probabilities = actions.data.numpy()[0]
            return np.random.choice(len(action_probabilities),p=action_probabilities)
    
    def choose_action(self, obs):
        """Method used in the testing process. Desired action is always the one with highest probability.

        Args:
            obs (array): current observation

        Returns:
            integer: the action taken by the agent in the next step
        """
        actions = self.network(torch.FloatTensor([obs]))
        actions = self.sm(actions)

        print(str(obs)+' --> '+str(actions))
        return torch.argmax(actions, dim=1).item()

    def train(self, elite_batch, obs, act):
        self.optimizer.zero_grad()
        action_scores = self.network(torch.FloatTensor(obs))
        loss_v = self.objective_func(action_scores,torch.LongTensor(act))
        loss_v.backward()
        self.optimizer.step()

        return loss_v.item()
    
    def load_pretrained_model(self, model_path="model/grab-coin-cross_entropy.pth"):
        self.network.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="model/grab-coin-cross_entropy.pth"):
        torch.save(self.network.state_dict(), model_path)