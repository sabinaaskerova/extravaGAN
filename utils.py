import torch
import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.mixture import GaussianMixture


def sample_gaussian_mixture(batch_size, n_components=10, latent_dim=100):
    """Sample from a Gaussian Mixture Model for the latent space"""
    # Randomly select which Gaussian to sample from
    component_indices = torch.randint(0, n_components, (batch_size,))
    
    # Generate random means and covariances for each component
    means = torch.randn(n_components, latent_dim) * 2
    covs = torch.exp(torch.randn(n_components, latent_dim)) * 0.5
    
    # Sample from selected Gaussians
    samples = torch.zeros(batch_size, latent_dim)
    for i in range(n_components):
        mask = (component_indices == i)
        n_samples = mask.sum().item()
        if n_samples > 0:
            samples[mask] = torch.normal(
                means[i].unsqueeze(0).repeat(n_samples, 1),
                covs[i].unsqueeze(0).repeat(n_samples, 1)
            )
    return samples

def D_train(x, G, D, D_optimizer, criterion):
    D.zero_grad()
    
    # Train on real data
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    
    # Train on fake data with mixture model sampling
    z = sample_gaussian_mixture(x.shape[0]).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()
    D_output = D(x_fake.detach())  # detach to avoid training G
    D_fake_loss = criterion(D_output, y_fake)
    
    # Gradient penalty for improved stability
    alpha = torch.rand(x.shape[0], 1).cuda()
    interpolated = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
    d_interpolated = D(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(d_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
    
    # Combined loss
    D_loss = D_real_loss + D_fake_loss + gradient_penalty
    D_loss.backward()
    D_optimizer.step()
    
    return D_loss.item()

def G_train(x, G, D, G_optimizer, criterion):
    G.zero_grad()
    
    # Generate fake data using mixture model sampling
    z = sample_gaussian_mixture(x.shape[0]).cuda()
    fake_data = G(z)
    
    # Multiple discriminator evaluations for feature matching
    d_fake = D(fake_data)
    
    # Use feature matching loss
    d_real = D(x.cuda())
    feature_matching_loss = torch.mean(torch.abs(torch.mean(d_real, dim=0) - torch.mean(d_fake, dim=0)))
    
    # Standard GAN loss
    labels_real = torch.ones(x.shape[0], 1).cuda()
    g_loss = criterion(d_fake, labels_real)
    
    # Combined loss
    total_g_loss = g_loss + 0.1 * feature_matching_loss
    total_g_loss.backward()
    G_optimizer.step()
    
    return total_g_loss.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
