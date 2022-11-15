"""
This folder contains models for trying out the DeepONet Architecture. 

"""
from turtle import down
import torch
from torch import nn
import numpy as np

def nonlinearity(option):
    """Returns the nonlinearity used in the various models. 
    
    Its worth noting, in pytorch, nonlinearities are implemented as
    classes. So, they need to initialized. Thats why you get this ()
    on the end of things. 

    Args:
        option (_type_): String indicated desired function. 

    Raises:
        NotImplementedError: If your passed string isn't accepted. 

    Returns:
        _type_: the nonlinearity from pytorch
    """
    if option.lower() == 'relu':
        return nn.ReLU()
    elif option.lower() == 'prelu':
        return nn.PReLU()
    elif option.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f'{option} is not supported yet.')


class DeepONet(nn.Module):
    """
    The wrapper method for the DeepONet. 
    """
    def __init__(self, branch_network, trunk_network, output_dim=1):
        """
        
        :param branch_network: a nn.Module containing the branch network
        :param trunk_network: a nn.Module containing the trunk network
        :param internal_dim: the dimension of the output of the branch
                             and trunk networks
        :param output_dim: the output dimension of the learned function itself.
        """
        super(DeepONet, self).__init__()
        assert output_dim == 1, "Only 1 dimensional outputs are supported."
        assert trunk_network.activate_last
        self.branch = branch_network
        self.trunk = trunk_network
        self.register_parameter(name='bias', param=nn.Parameter(torch.randn(1)))

    def forward(self, x, u):
        """
        :param x: the x coordinates of interest for the model. 
        :param u: the parameterizing u function for the operator.
        """
        b_vec = self.branch(u)
        t_vec = self.trunk(x)
        return (b_vec * t_vec).sum(dim=-1).unsqueeze(-1) + self.bias

class Shallow(nn.Module):
    """
    A single fully connected layer. 
    """
    def __init__(self, inputs, outputs, activation='relu', activate_last=True):
        """
        inputs: the number of input dimensions
        outputs: the number of output dimensions
        """
        super(Shallow, self).__init__()
        self.activate_last = activate_last
        if activate_last:
            self.model = nn.Sequential(
                    nn.Linear(inputs, outputs),
                    nonlinearity(activation)
                    )
        else:
            self.model = nn.Linear(inputs, outputs)
    
    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    """
    A dense linear network
    """
    def __init__(self, layer_dimensions, activation='relu', activate_last=True):
        """
        :param layer_dimensions: a list of all of the intermediate dimensions
        """
        super(MLP, self).__init__()
        self.activate_last = activate_last
        model = []
        for i in range(len(layer_dimensions)-1):
            model.append(nn.Linear(layer_dimensions[i], layer_dimensions[i+1]))
            if activate_last or i != len(layer_dimensions)-2:
                model.append(nonlinearity(activation))
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


def conv1_module(in_channels, out_channels, half_size=True, kernel_size=3):
    """
    A method for defining convolutional modules
    """
    stride = 2 if half_size else 1
    assert (kernel_size >= 3) and (kernel_size <= 5), "Kernel Size must be between 3 and 5."

    k_constant = (kernel_size - 1) / 2
    if int(k_constant) != k_constant:
        padding = int(k_constant)
    else:
        padding = int(k_constant) - 1

    conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    return conv_layer

class VectorizedCNN(nn.Module):
    """
    A simple CNN (1 dimensional) with a vectorized output. 

    In this implementation, I am going to give the user much
    more flexibility. I am also going to use global mean pooling.
    The output vector is just going to be the channels at the end. 
    """
    def __init__(self, 
                 features=(1, 32, 16, 8, 8),
                 kernel_sizes=(5, 3, 3, 3), 
                 down_size=(True, True, True, False),
                 activation='relu', 
                 activate_last=False,
                ):
        """
        :param features: a tuple/list containing the number of features in each layer. 
                         The length of this object also defines the depth of the
                         network. Should include the initial size as well. 
        :param kernel_sizes: A list of kernel sizes equal in length to the
                             feature list. 
        :param down_size: a list/tuple of boolians indicating whether this layer
                          should decrease the size. 
        :param activation: the nonlinear activation
        :param activate_last: a boolian indicating whether to activate the last output. 
        """
        super(VectorizedCNN, self).__init__()
        assert len(features) == len(kernel_sizes)+1, "Number of layers must be constant."
        assert len(down_size) == len(kernel_sizes), "Number of layers must be constant."

        self.num_layers = len(down_size)
        for n, down_boolian in enumerate(down_size):
            conv_layer = conv1_module(
                in_channels=features[n],
                out_channels=features[n+1],
                half_size=down_boolian,
                kernel_size=kernel_sizes[n]
            )

            setattr(self, f"conv{n}", conv_layer)

        self.nonlinear = nonlinearity(activation)
        self.activate_last = activate_last

    def forward(self, x):
        """
        the forward method.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        if self.activate_last:
            # then the last layer gets activation
            for i in range(0, self.num_layers):
                layer = getattr(self, f"conv{i}")
                x = self.nonlinear(layer(x))
            # global pooling
            x = x.mean(dim=-1)
            return x
        else:
            # then the last layer does not
            for i in range(0, self.num_layers-1):
                layer = getattr(self, f"conv{i}")
                x = self.nonlinear(layer(x))
            layer = getattr(self, f"conv{self.num_layers-1}")
            x = layer(x)
            x = x.mean(dim=-1)
            return x
