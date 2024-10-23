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
    parser = argparse.ArgumentParser(description='Train Improved GAN')
    parser.add_argument("--epochs", type=int, default=200,
                      help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="Size of mini-batches for SGD")
    
    args = parser.parse_args()
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])
    
    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=4)
    
    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()
    
    # Optimizers with better hyperparameters
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    print('Start Training:')
    
    for epoch in trange(1, args.epochs + 1):
        G.train()
        D.train()
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            
            # Train discriminator
            d_loss = D_train(x, G, D, D_optimizer, criterion)
            
            # Train generator
            g_loss = G_train(x, G, D, G_optimizer, criterion)
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'd_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
        
        # Save models periodically
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
            
    print('Training completed')