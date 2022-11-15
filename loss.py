"""
A file containing all the loss funcitons
"""
import torch
import torch.nn as nn
import utils

def standard_loss(y, u, s, xlocations, model, device):
    """ 
    this is the standard DeepONet loss 

    y: the location we are estimating s
    u: the parameterizing function (in this case, the derivative function)
    s: the true value of s
    xlocations: the xlocations where we have sampled s
    """
    # computing the true s values
    with torch.no_grad():
        real_s = torch.zeros_like(y)
        for n, s_individual in enumerate(s):
            func = utils.interpolated_function(xlocations, s_individual)
            real_s[n] = func(y[n])
    
    y = y.to(device)
    real_s = real_s.to(device)
    
    return torch.square(model(y, u) - real_s).mean()

def pinn_loss(u, x, s, model):
    """ this is the standard DeepONet Loss, augmented with a PINN term """
    raise NotImplementedError
    # computing the true s and u values
    with torch.no_grad():
        real_s = torch.zeros_like(y)
        real_u = torch.zeros_like(y)
        for n, s_individual in enumerate(s):
            func_s = utils.interpolated_function(xlocations, s)
            func_u = utils.interpolated_function(xlocations, s)
            real_s[n] = func_s(y[n])
            real_u = func_u(y[n])
    
    return torch.square(model(y, u) - real_s)
