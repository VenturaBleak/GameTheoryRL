import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-network

    Args:
        input_size (int): Number of input features
        hidden_sizes (list): List of hidden layer sizes, e.g. 5 layers: [10, 20, 30, 40, 50]
        num_actions (int): Number of actions
        """
    def __init__(self, input_size, hidden_sizes, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.layers.append(nn.Linear(hidden_sizes[-1], num_actions))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)  # No activation on the last layer