"""
For testing code
"""
import matplotlib.pyplot as plt
import torch
import utils
import dataset
import models


def test_interpolation():
    x = torch.linspace(0, 1, 20)
    y = torch.square(x)

    func = utils.interpolated_function(x, y)

    xprime = torch.linspace(0, 1, 100)

    output = utils.integrate(xprime, func)

    f, ax = plt.subplots(1, 2)
    ax[0].plot(x, y)
    ax[1].plot(xprime, output)
    ax[1].plot(x, (1/3) * (x ** 3), 'ro')
    plt.show()

def test_and_createdataset():
    #dataset.create_dataset()

    uxlocation, data = dataset.load_dataset('./data/train.pth')
    
    #print(uxlocation.shape)
    #[print(item.shape) for item in data]

    DONDataset = dataset.DeepONetDataset(data)
    dataloader = torch.utils.data.DataLoader(DONDataset, 
                                             batch_size=64, 
                                             shuffle=True,
                                             )

    for n, (us, ss, y) in enumerate(dataloader):
        print('-'*40)
        print('Enumerating')
        print(us.shape)
        print(ss.shape)
        print(y.shape)
        f, ax = plt.subplots(1, 2)
        for i in range(len(us)):
            ax[0].plot(uxlocation, us[i])
            ax[1].plot(uxlocation, ss[i])
        plt.show()
        exit()
        



def test_models():
    mlp = models.MLP([1, 10, 9, 8], activate_last=True)
    x = torch.rand(30, 1)

    print('-'*40)
    print('Testing MLP')
    print(mlp)
    print(mlp(x).shape)

    # CNN
    print('-'*40)
    print('Testing CNN')
    cnn = models.VectorizedCNN(
                 features=(1, 32, 16, 8, 8),
                 kernel_sizes=(5, 3, 3, 3), 
                 down_size=(True, True, True, False),
                 activation='relu', 
                 activate_last=False,
    )

    u_samples = torch.rand(30, 100)
    embedded = cnn(u_samples)
    print(embedded.shape)
    print(embedded.max())
    print(embedded.min())

    # Testing the DeepONet
    print('-'*40)
    print("Testing DeepONet")
    deepo = models.DeepONet(cnn, mlp)

    output = deepo(x, u_samples)
    print(output.shape)
    print(deepo)

    print(deepo.bias)



def main():
    x = torch.linspace(0, 1, 100)
    samples = 10
    l = 0.3
    funcs = utils.gp_prior(x, l=l).sample(samples)

    f, ax = plt.subplots(1, 2)
    for i in range(samples):
        ax[0].plot(x.numpy(), funcs[i].numpy(), 'k-')

    plt.show()

if __name__ == "__main__":
    #test_interpolation()
    #test_models()
    #main()
    test_and_createdataset()

    
