import torch.nn as nn

class Network(nn.Module):
    def __init__(self,obs_size,hidden_size,num_actions):
        super(Network,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,num_actions)
        )
    def forward(self,x):
        return self.network(x)