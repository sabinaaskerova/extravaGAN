import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

def sample_gmm(batch_size, n_components=8, latent_dim=100):
    """Sample from a Gaussian Mixture Model instead of standard normal distribution"""
    # Select which Gaussian to sample from
    component_indices = torch.randint(0, n_components, (batch_size,))
    
    # Define means for each component (arranged in a circle)
    thetas = torch.arange(n_components) * (2 * np.pi / n_components)
    radius = 2
    means = torch.stack([radius * torch.cos(thetas), radius * torch.sin(thetas)], dim=1)
    
    # Extend means to full latent dimension
    means = torch.cat([means, torch.zeros(n_components, latent_dim-2)], dim=1).cuda()
    
    # Sample from selected Gaussians
    samples = torch.randn(batch_size, latent_dim).cuda()
    for i in range(batch_size):
        samples[i] = samples[i] + means[component_indices[i]]
    
    return samples

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()
    
    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()
    
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output
    
    # train discriminator on fake
    # Use GMM instead of standard normal distribution
    z = sample_gmm(x.shape[0])
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()
    
    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output
    
    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    
    return D_loss.data.item()

def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()
    
    # Use GMM instead of standard normal distribution
    z = sample_gmm(x.shape[0])
    y = torch.ones(x.shape[0], 1).cuda()
    
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)
    
    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
    
    return G_loss.data.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))

def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
