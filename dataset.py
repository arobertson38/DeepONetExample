"""
This file contains any necessary tools for creating and
loading datasets
"""
import torch
import utils
import pickle
import os

def _dataset_creator(l_param=0.3, usamples=100, \
        uxlocation=torch.linspace(0, 1, 100), ysamples=100):
    """ this method does the leg work for create_dataset """
    us = utils.gp_prior(uxlocation, l=l_param).sample(usamples)
    ysamples = torch.rand(ysamples, 1)

    # evaluating the goal functions
    ss = torch.zeros_like(us)
    for i in range(len(ss)):
        func = utils.interpolated_function(uxlocation, us[i])
        ss[i] = utils.integrate(uxlocation, func)

    return [uxlocation, us, ss, ysamples]

def create_dataset(l_param=0.3, usamples=120, \
        uxlocation=torch.linspace(0, 1, 100), ysamples=120, \
        testmultiplier=0.2, save_location='./data/'):
    """
    This method creates a training and a testing dataset. 
    
    l_param: the lengthscale parameter of the Gaussian
             Random Field. 
    usamples: the number of gaussian random field samples
    uxlocation: the x location of the GRF samples
    ysamples: the number of spatial location samples
    """
    assert testmultiplier >= 0 and testmultiplier <= 1
    train_dataset = _dataset_creator(l_param, usamples, \
            uxlocation, ysamples)

    test_dataset = _dataset_creator(l_param, int(usamples * testmultiplier), \
            uxlocation, int(ysamples * testmultiplier))

    if save_location is not None:
        with open(os.path.join(save_location, 'train.pth'), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(save_location, 'test.pth'), 'wb') as f:
            pickle.dump(test_dataset, f)

    else:
        return train_dataset, test_dataset

def load_dataset(filename):
    """ 
    a wrapper to load the files 

    first output is the x locations where the parameterizing function
    is computed. The second output is a list:

        [us, ss, ysamples]

    """
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset[0], dataset[1:]

# -----------------------------------------------
# Torch dataset object
# -----------------------------------------------

class DeepONetDataset(torch.utils.data.Dataset):
    """
    A custom dataset loading object. The primary utility of this is to
    repeat the u_samples and s_samples for multiple calls to the ysamples
    without having to store everything in one horribly large array. 
    """
    def __init__(self, data_list):
        """
        data_list: a list containing the stored data. 
                   [us, ss, ysamples]
        """
        self.us, self.ss, self.ysample = data_list
        self.length = len(self.us) * len(self.ysample)

    def split_index(self, index):
        """ splits the index into a u and y index """
        u_index = int(index / len(self.us))
        y_index = int(index % len(self.us))
        return [u_index, y_index]

    def __len__(self):
        return self.length

    def __getitem__(self, indx):
        # OLD CODE FOR MULTI INDEXING
        #if torch.is_tensor(indx):
        #    indx = indx.tolist()
        #else:
        #    indx = [indx, ]
        #index = torch.tensor(list(map(self.split_index, indx)))
        #return self.us[index[:, 0]], \
        #        self.ss[index[:, 0]], self.ysample[index[:, 1]]
        assert type(indx) is int, "Only suppport single indexing."
        func_indx, y_indx = self.split_index(indx)
        return self.us[func_indx], self.ss[func_indx], self.ysample[y_indx]
        












