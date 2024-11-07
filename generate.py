import torch 
import torchvision
import os
import argparse
import random
import numpy as np


from model import Generator, Discriminator
from utils import load_model
import torch.nn.functional as F

def BurnIn(G, D_star, num_samples=1000):
    max_ratio = 0
    for _ in range(num_samples):
        z = torch.randn(1, 100).cuda()
        x = G(z)
        ratio = torch.exp(D_star(x)).item()
        max_ratio = max(max_ratio, ratio)
    return max_ratio

def DiscriminatorRejectionSampling(G, D, z_batch, epsilon=0.001):
    D_star = D
    D_star_M = BurnIn(G, D_star)
    M_bar = D_star_M
    
    samples = []

    # Compute F(x) values for the entire batch
    F_values = []
    for z in z_batch:
        x = G(z.unsqueeze(0))  # Generate a single sample
        ratio = torch.exp(D_star(x)).item()
        M_bar = max(M_bar, ratio)
        
        # Calculate F(x) without gamma for now and store it
        F = D_star(x) - M_bar - torch.log(1 - torch.exp(D_star(x) - M_bar - epsilon))
        F_values.append(F)

    # Stack F_values to a tensor
    F_values = torch.stack(F_values)

    # Calculate gamma as the 95th percentile of F_values (sorting and indexing)
    gamma = torch.quantile(F_values, 0.95)

    # Loop again for sampling decision with gamma
    for i, z in enumerate(z_batch):
        x = G(z.unsqueeze(0))
        F = F_values[i] - gamma  # Subtract gamma from stored F(x)
        p = torch.sigmoid(F).item()

        psi = random.uniform(0, 1)
        if psi <= p:
            samples.append(x.squeeze())
    
    return samples


def generate_samples_with_drs(G, D, batch_size):
    z_batch = torch.randn(batch_size, 100).cuda()
    samples = DiscriminatorRejectionSampling(G, D, z_batch)
    return torch.stack(samples)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    G = Generator(g_output_dim = mnist_dim).cuda()
    G = load_model(G, 'checkpoints', 'G.pth')
    G = torch.nn.DataParallel(G).cuda()
    G.eval()

    D = Discriminator(d_input_dim=mnist_dim).cuda()
    D = load_model(D, 'checkpoints', 'D.pth')
    D = torch.nn.DataParallel(D).cuda()
    D.eval()

    print('Model loaded.')


    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            x = generate_samples_with_drs(G, D, args.batch_size)
            x = x.reshape(-1, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1


    
