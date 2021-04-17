import torch.nn as nn

class Network(nn.Module):
    def __init__(self,layer_sizes):
        super(Network,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(layer_sizes[0],layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1],layer_sizes[2])
        )
        
    def forward(self,x):
        return self.network(x)