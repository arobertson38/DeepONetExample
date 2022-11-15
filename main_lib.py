"""
This file contains the main contents of the code. 

Created by Andreas
"""
import torch
import dataset
import loss
import utils
import numpy as np
import matplotlib.pyplot as plt


def test():
    """
    A method for testing trained models
    """
    # Parameters
    model_name = 'test1.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Lazily loading the data for no reason
    xlocations, data = dataset.load_dataset('./data/test.pth')
    test_dataset = dataset.DeepONetDataset(data)
    test_dataloader = torch.utils.data.DataLoader( \
            test_dataset, batch_size=128, shuffle=False)

    # loading the model
    model = utils.get_model()
    model = utils.load_model(model, 'test1.pth').to(device)
    model.eval()

    # plotting the first result
    for n, (us, ss, ys) in enumerate(test_dataloader):
        f, ax = plt.subplots(1, 2)
        ax[0].set_title('The Function Value')
        ax[1].set_title('Its Derivative Value')
        ax[0].plot(xlocations.numpy(), ss[n].numpy())
        ax[1].plot(xlocations.numpy(), us[n].numpy())

        with torch.no_grad():
            xlocs = xlocations.unsqueeze(-1).to(device)
            u = torch.cat([us[n].unsqueeze(0)] * len(xlocs), dim=0).to(device)
            predicted_func = model(xlocs, u).to('cpu')
        
        ax[0].plot(xlocations.numpy(), predicted_func.squeeze().detach().numpy(), 'r-')
        
        plt.show()
    
    
    
    

def train():
    # parameters
    model_name = 'test1.pth'
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initializing the model 
    model = utils.get_model()
    model.to(device)

    # loading the dataset
    xlocations, data = dataset.load_dataset('./data/train.pth')
    train_dataset = dataset.DeepONetDataset(data)
    train_dataloader = torch.utils.data.DataLoader( \
            train_dataset, batch_size=128, shuffle=True)

    # loss function
    loss_fn = loss.standard_loss

    # optimizer
    optim = utils.get_optimizer(model.parameters())

    # running
    model.train()
    losses = []
    for epoch in range(epochs):
        for _, (us, ss, ys) in enumerate(train_dataloader):
            # moving everything to device
            us = us.to(device)
            #ss = ss.to(device)
            # ys = ys.to(device) # ys isn't moved to the device because we need
            # to do a bunch of weird operations to it. So, avoiding doing this
            # on the GPU. 
            
            # training inner loop
            optim.zero_grad()
            error = loss_fn(ys, us, ss, xlocations, model, device)
            error.backward()
            optim.step()

        print(f"Epoch: {epoch}. Loss={error.item()}")
        losses.append(error.item())

    utils.save_model(model, model_name)

    # plotting the loss curves
    f, ax = plt.subplots(1, 1)
    ax.plot(np.arange(epochs), np.array(losses), 'k-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.show()






if __name__ == "__main__":
    test()
    exit()
    train()
