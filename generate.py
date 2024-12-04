import torch 
import torchvision
import os
import argparse
import random



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

def DiscriminatorRejectionSampling(G, D, N, epsilon, gamma):
    D_star = D
    D_star_M = BurnIn(G, D_star)
    M_bar = D_star_M
    
    samples = []
    
    while len(samples) < N:
        z = torch.randn(1, 100).cuda()
        x = G(z)
        
        # Compute acceptation
        ratio = torch.exp(D_star(x)).item()
        M_bar = max(M_bar, ratio)
        
        # Compute probability of acceptation
        F = (D_star(x) - D_star_M
                   - torch.log(1 - torch.exp(D_star(x) - D_star_M - epsilon))
                   - gamma)
        p = torch.sigmoid(F).item()
        
        # We decide if we accept it or not
        psi = random.uniform(0, 1)
        if psi <= p:
            samples.append(x.squeeze())
    
    return samples

def generate_samples_with_drs(z):
    D = Discriminator(d_input_dim=784).cuda()
    D = load_model(D, 'checkpoints', 'D.pth')
    D = torch.nn.DataParallel(D).cuda()
    D.eval()

    # get sample with the dicriminator
    samples = DiscriminatorRejectionSampling(G, D, len(z), epsilon=1e-10, gamma=0.001)
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
    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).cuda()
            x = generate_samples_with_drs(z) 
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1


    
