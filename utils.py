import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import os

# Define auxiliary function to add noise
def add_noise(z, sigma_noise=0.03):
    noise = torch.normal(0, sigma_noise, z.shape).cuda()
    return z + noise

# D_train function with bCR and zCR regularization
def D_train(x, G, D, D_optimizer, criterion, noise_factor=0.03, lambda_real=10, lambda_fake=10, lambda_dis=5):
    D.zero_grad()
    
    # Real images
    real_label = torch.ones(x.size(0), 1).cuda()
    D_x = D(x)
    errD_real = criterion(D_x, real_label)
    
    # bCR for real images: augment x and calculate consistency loss
    T_x = transforms.RandomHorizontalFlip()(x)
    D_T_x = D(T_x)
    L_real = F.mse_loss(D_x, D_T_x)
    
    # Total loss for real images
    real_loss = errD_real + lambda_real * L_real
    real_loss.backward()
    
    # Fake images
    z = torch.randn(x.size(0), 100).cuda()
    z = add_noise(z, sigma_noise=noise_factor)  # zCR transformation
    G_z = G(z)
    fake_label = torch.zeros(x.size(0), 1).cuda()
    D_G_z = D(G_z.detach())
    errD_fake = criterion(D_G_z, fake_label)
    
    # bCR for fake images: augment generated images and calculate consistency loss
    T_G_z = transforms.RandomHorizontalFlip()(G_z.detach())
    D_T_G_z = D(T_G_z)
    L_fake = F.mse_loss(D_G_z, D_T_G_z)
    
    # zCR for discriminator: forward transformed z through generator and calculate consistency loss
    T_z = add_noise(z, sigma_noise=noise_factor)
    G_T_z = G(T_z)
    D_G_T_z = D(G_T_z.detach())
    L_dis = F.mse_loss(D_G_z, D_G_T_z)
    
    # Total loss for fake images
    fake_loss = errD_fake + lambda_fake * L_fake + lambda_dis * L_dis
    fake_loss.backward()
    
    D_optimizer.step()
    
    return (real_loss + fake_loss).item()

# G_train function with zCR regularization for generator
def G_train(x, G, D, G_optimizer, criterion, noise_factor=0.03, lambda_gen=0.5):
    G.zero_grad()
    
    # Generate fake images
    z = torch.randn(x.size(0), 100).cuda()
    G_z = G(z)
    
    # Label for generator training (trying to fool the discriminator)
    real_label = torch.ones(x.size(0), 1).cuda()
    D_G_z = D(G_z)
    errG = criterion(D_G_z, real_label)
    
    # zCR for generator: transformed z consistency loss
    T_z = add_noise(z, sigma_noise=noise_factor)
    G_T_z = G(T_z)
    L_gen = -F.mse_loss(G_z, G_T_z)  # zCR regularization for generator

    # Total generator loss
    generator_loss = errG + lambda_gen * L_gen
    generator_loss.backward()
    
    G_optimizer.step()

    return generator_loss.item()

def save_models(G, D, folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder, 'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
