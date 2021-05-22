import torch.nn as nn

class Network(nn.Module):
    def __init__(self,layer_sizes):
        """The network consists of three layers. The activation function used in the hidden layer is ReLu. Input size is based on the observation size. Similarly, the size of the output depends on the number of actions. 

        Args:
            layer_sizes (tuple): sizes of individual network layers
        """
        super(Network,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(layer_sizes[0],layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1],layer_sizes[2])
        )
        
    def forward(self,x):
        return self.network(x)