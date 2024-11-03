import torch
from torch import nn
import os
import numpy as np
# possible augmentations: image flipping and rotating, sentence back-translating, etc.
def add_noise(inputs, noise_factor=0.1):
    noise = noise_factor * torch.randn_like(inputs)
    return inputs + noise

def D_train(x, G, D, D_optimizer, criterion, noise_factor=0.1):
    D.zero_grad()
    # Train discriminator on real
    x_real, y_real = x.cuda(), torch.rand(x.shape[0], 1).cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z).detach()
    y_fake = torch.rand(x.shape[0], 1).cuda() * 0.2 # label smoothing
    D_output = D(x_fake)

    D_fake_loss = criterion(D_output, y_fake) 

    # Apply noise for consistency regularization
    x_fake_noisy = add_noise(x_fake, noise_factor) # augmented input to D
    D_output_noisy = D(x_fake_noisy)
    consistency_loss = criterion(D_output_noisy, y_fake)

    # Total loss
    D_loss = D_real_loss + D_fake_loss + consistency_loss # consistency regularization
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, noise_factor=0.1):
    G.zero_grad()
    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()

    G_output = G(z).cuda()
    D_output = D(G_output).cuda()

    # Adversarial loss
    G_adv_loss = criterion(D_output, y) 

    # Consistency regularization
    G_output_noisy = add_noise(G_output, noise_factor)
    D_output_noisy = D(G_output_noisy).cuda()
    
    G_consistency_loss = criterion(D_output_noisy, y)

    G_loss = G_adv_loss + 0.1 * G_consistency_loss  # Adjust the weight as needed

    # Gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def save_models(G, D, folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in  ckpt.items()})
    return G