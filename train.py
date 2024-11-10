import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator
from utils import D_train, G_train, save_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--noise_factor", type=float, default=0.1,
                        help="Noise factor for consistency regularization")
    parser.add_argument("--gname", type=str, default="G.pth",
                        help="File name for saving the generator model checkpoint.")
    parser.add_argument("--dname", type=str, default="D.pth",
                        help="File name for saving the discriminator model checkpoint.")
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()
    print('Model loaded.')



    # define loss
    # criterion = nn.BCELoss() 
    criterion = nn.MSELoss()

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            D_loss = D_train(x, G, D, D_optimizer, criterion, noise_factor=args.noise_factor)
            G_loss = G_train(x, G, D, G_optimizer, criterion, noise_factor=args.noise_factor)
            # D_loss = D_train(x, G, D, D_optimizer, criterion)
            # G_loss = G_train(x, G, D, G_optimizer, criterion)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints', gname=args.gname, dname=args.dname)

    print('Training done')