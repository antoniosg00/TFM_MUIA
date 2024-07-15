# PyTorch model and training necessities
import torch
import torch.nn as nn
import tensorboard as tb
import numpy as np
import random


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


def initialized_layer(layer, init='orthogonal', gain=np.sqrt(2)):
    """
    Initialize the weights and biases of a PyTorch layer.

    Args:
    - layer: PyTorch layer object.
    - init: Initialization method for weights. Options: 'orthogonal', 'normal', 'uniform'. Default is 'orthogonal'.
    - gain: Scaling factor for initialization. Default is sqrt(2).

    Returns:
    - layer: Initialized layer with weights and biases.
    """

    # Initialize weights based on the chosen initialization method
    if init == 'orthogonal':
        torch.nn.init.orthogonal_(layer.weight, gain)
    elif init == 'normal':
        torch.nn.init.xavier_normal_(layer.weight, gain)
    elif init == 'uniform':
        torch.nn.init.xavier_uniform_(layer.weight, gain)
    else:
        raise Exception('Initialization method must be "orthogonal", "normal" or "uniform".')

    # Initialize biases to zero
    torch.nn.init.constant_(layer.bias, 0.)

    return layer

class MainBlock(nn.Module):
    """
    MainBlock class defines the core structure of neural network layers.

    Args:
    - input_size: Size of the input layer.
    - hidden_sizes: List of sizes for hidden layers.
    - dropout_prob: Dropout probability. Default is 0.0 (no dropout).
    - activation: Activation function. Options: 'tanh', 'lrelu'. Default is 'tanh'.
    - lrelu: LeakyReLU slope parameter. Default is 0.01.
    - bn: Boolean indicating if BatchNormalization is used. Default is False.
    - momentum: Momentum for BatchNormalization. Default is 0.9.
    - initialization: Weight initialization method. Options: 'orthogonal', 'normal', 'uniform'. Default is 'normal'.
    """
    def __init__(self, input_size, hidden_sizes, dropout_prob=0.0, activation='tanh', lrelu=0.01, bn=False, momentum=0.9, initialization='normal', **kwargs):
        super(MainBlock, self).__init__()
        self.layers = nn.ModuleList()
        # Iterate through hidden_layer sizes to create neural network layers
        for i in range(len(hidden_sizes)):
            # Linear layer with initialization
            linear_layer = nn.Linear(input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i])
            self.layers.append(initialized_layer(linear_layer, init=initialization, gain=np.sqrt(2)))
            # BatchNormalization layer
            if bn:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i], momentum=1-momentum))  # (momentum_pytorch = 1 - momentum_tensorflow)
            # Activation layer (LeakyReLU or tanh)
            if activation == 'lrelu':
                self.layers.append(nn.LeakyReLU(lrelu))
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            else:
                raise Exception('activation parameter must be "tanh" or "lrelu".')
            # Dropout layer
            if dropout_prob > 0:
                self.layers.append(nn.Dropout(dropout_prob))
    

    def forward(self, x):
        """
        Forward pass through the MainBlock layers.

        Args:
        - x: Input tensor.

        Returns:
        - x: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)

        return x


class Actor(nn.Module):
    """
    Actor class represents the policy network for the PPO algorithm.

    Args:
    - input_size: Size of the input state.
    - hidden_sizes: List of sizes for hidden layers.
    - output_size: Size of the output (number of actions).
    - dropout_prob: Dropout probability. Default is 0.0 (no dropout).
    - activation: Activation function. Options: 'tanh', 'lrelu'. Default is 'tanh'.
    - lrelu: LeakyReLU slope parameter. Default is 0.01.
    - bn: Boolean indicating if BatchNormalization is used. Default is False.
    - momentum: Momentum for BatchNormalization. Default is 0.95.
    - initialization: Weight initialization method. Options: 'orthogonal', 'normal', 'uniform'. Default is 'orthogonal'.
    """
    DEFAULTS = {
    'input_size': 8,
    'hidden_sizes': [256, 128, 64],
    'output_size': 6,  # Number of actions for the actor
    'dropout_prob': 0.0,  # Dropout rate for the Dropout layer
    'activation': 'tanh',  # Activations for Linear layers
    'lrelu': 0.01,  # Optional parameter for the LeakyReLU activation function
    'bn': False, # If there are BatchNormalization layers
    'momentum': 0.95,  # Momentum value for the BatchNormalization layer
    'initialization': 'orthogonal',  # Weight initializer for the model layers
    }
    def __init__(self, **kwargs):
        super(Actor, self).__init__()
        # Apply default values for missing parameters
        params = {**self.DEFAULTS, **kwargs}
        self.params_test = params

        self.main_block = MainBlock(**params)
        # Initialize output layer for actor
        linear_layer = nn.Linear(params['hidden_sizes'][-1], params['output_size'])
        self.output_layer = initialized_layer(linear_layer, init='orthogonal', gain=0.01)
        self.softmax = nn.Softmax(dim=-1)
    

    def forward(self, x):
        """
        Forward pass through the Actor network.

        Args:
        - x: Input state tensor.

        Returns:
        - x: Output probability distribution over actions.
        """
        x = self.main_block(x)
        x = self.output_layer(x)
        x = self.softmax(x)  # Tensor of probabilities

        return x


    def get_action(self, state):
        """
        Sample an action from the Actor's policy distribution.

        Args:
        - state: Input state tensor.

        Returns:
        - action: Sampled action.
        """
        # Get the categorical distribution for the current state
        dist = torch.distributions.Categorical(self.forward(state))
        # Sample an action
        action = dist.sample()

        return action


class Critic(nn.Module):
    """
    Critic class represents the value network for the PPO algorithm.

    Args:
    - input_size: Size of the input state.
    - hidden_sizes: List of sizes for hidden layers.
    - output_size: Size of the output (1 for value estimation).
    - dropout_prob: Dropout probability. Default is 0.0 (no dropout).
    - activation: Activation function. Options: 'tanh', 'lrelu'. Default is 'tanh'.
    - lrelu: LeakyReLU slope parameter. Default is 0.01.
    - bn: Boolean indicating if BatchNormalization is used. Default is False.
    - momentum: Momentum for BatchNormalization. Default is 0.95.
    - initialization: Weight initialization method. Options: 'orthogonal', 'normal', 'uniform'. Default is 'normal'.
    """
    DEFAULTS = {
    'input_size': 6,
    'hidden_sizes': [256, 128, 64],
    'output_size': 6,  # Number of actions for the actor
    'dropout_prob': 0.0,  # Dropout rate for the Dropout layer
    'activation': 'tanh',  # Activations for Linear layers
    'lrelu': 0.01,  # Optional parameter for the LeakyReLU activation function
    'bn': False, # If there are BatchNormalization layers
    'momentum': 0.95,  # Momentum value for the BatchNormalization layer
    'initialization': 'normal',  # Weight initializer for the model layers
    }
    def __init__(self, **kwargs):
        super(Critic, self).__init__()
        # Apply default values for missing parameters
        params = {**self.DEFAULTS, **kwargs}
        self.params_test = params

        self.main_block = MainBlock(**params)
        # Initialize output layer for critic
        linear_layer = nn.Linear(params['hidden_sizes'][-1], 1)
        self.output_layer = initialized_layer(linear_layer, init='orthogonal', gain=1)
    

    def forward(self, x):
        """
        Forward pass through the Critic network.

        Args:
        - x: Input state tensor.

        Returns:
        - x: Estimated value of the input state.
        """
        x = self.main_block(x)
        
        return self.output_layer(x)
    
