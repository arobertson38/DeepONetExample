"""
Utility methods for this. 

"""
import torch
from functools import partial
import numpy as np
import os
import models

#-----------------------------------------------
# Getting Utils. 
#-----------------------------------------------

def get_model(configs=None):
    """ returns the model identified by the configs """
    trunk = models.MLP([1, 10, 9, 8], activate_last=True)
    branch = models.VectorizedCNN(
                 features=(1, 32, 16, 8, 8),
                 kernel_sizes=(5, 3, 3, 3), 
                 down_size=(True, True, True, False),
                 activation='relu', 
                 activate_last=False,
    )
    return models.DeepONet(branch, trunk)

def get_optimizer(parameters, configs=None):
    """ returns the optimizer identified by the config file """
    return torch.optim.Adam(parameters, \
            lr=0.001, \
            betas=(0.9, 0.999), \
            )

#-----------------------------------------------
# Saving and Loading
#-----------------------------------------------

def save_model(model, model_name='test1.pth', locations='./models'):
    """
    Saving the library of a pytorch model. 

    Args:
        model_name (str, optional): _description_. Defaults to 'test1.pth'.
        locations (str, optional): _description_. Defaults to './models'.
    """
    torch.save(model.state_dict(), \
        os.path.join(locations, model_name))

def load_model(model, model_name='test1.pth', locations='./models'):
    """
    loading and initializing from a saved model. 

    Args:
        model (_type_): _description_
        model_name (str, optional): _description_. Defaults to 'test1.pth'.
        locations (str, optional): _description_. Defaults to './models'.
    """
    model.load_state_dict(
        torch.load(os.path.join(locations, model_name))
    )
    return model

#-----------------------------------------------
# Gaussian Priors
#-----------------------------------------------

def covariance_factory(setting='se'):
    """
    a factory for covariance functions
    """
    if setting.lower() == 'se':
        def covariance(r, l=1):
            return torch.exp(-1 * torch.square(r) / (2 * l**2))
    
    else:
        raise NotImplementedError(f"{setting} is not a supported covariance kernel.")
    return covariance

def covariance_matrix(x, cov_function, eps=1e-5):
    """
    a method that builds covariance matrices

    assumes teh covariance function is a partial function
    """
    v1, v2 = torch.meshgrid(x, x)
    return cov_function(v1-v2) + eps * torch.eye(len(x))


class gp_prior(object):
    """
    a gp_prior
    """
    def __init__(self, x_points, setting='se', **kwargs):
        """
        x_points: the x locations to sample at
        setting: the covariance function
        kwargs: any keyword arguments for the covariance function
        """
        assert type(x_points) == torch.Tensor
        assert len(x_points.shape) == 1
        cov_function = partial(covariance_factory(setting), **kwargs)
        cov_matrix = covariance_matrix(x_points, cov_function)

        self.L = torch.linalg.cholesky(cov_matrix)
        self.dimension = len(x_points)

    def sample(self, n_samples=1):
        """
        sampling the GP prior
        """

        samples = torch.randn(self.dimension, n_samples)
        return (self.L @ samples).transpose(0, 1)


def interpolated_function(x, y):
    """ creates a function by linearly interpolating between points in x """
    assert len(x.shape) == 1
    def function(t_vec, *args):
        """ the returned function """
        def internal_func(t):
            assert t >= x[0] and t <= x[-1]
            index = ((x - t) >= 0).float().argmax()
            if index != 0:
                index -= 1
            
            x1 = x[index]
            x2 = x[index + 1]
            y1 = y[index]
            y2 = y[index + 1]

            if type(x) == torch.Tensor:
                return ((y2 - y1) / (x2 - x1) * (t - x1) + y1).item()
            else:
                return (y2 - y1) / (x2 - x1) * (t - x1) + y1

        if (type(t_vec) is float) or (type(t_vec) is np.float64):
            return internal_func(t_vec)
        else:
            y_output = list(map(internal_func, t_vec))
            if type(t_vec) == torch.Tensor:
                return torch.tensor(y_output)
            else:
                return np.array(y_output)


    return function



#-----------------------------------------------
# Gaussian Priors
#-----------------------------------------------

def integrate(x, dsdx, interval=(0.0, 1.0), \
        initial_condition=torch.tensor([0.0])):
    """
    integrates assuming the initial condition holds at the first
    x
    dsdx: a method that returns the derivative of s
    with respect to x
    """
    from scipy.integrate import solve_ivp

    output = solve_ivp( \
            fun=dsdx, \
            t_span = interval, \
            y0 = initial_condition, \
            t_eval = x, \
            )

    return torch.from_numpy(output.y).squeeze()
    









