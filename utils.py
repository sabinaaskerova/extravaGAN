import torch
import os
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np

'''
def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.rand(x.shape[0], 1) * 0.2 + 0.8
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.rand(x.shape[0], 1).cuda() * 0.2 

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()
'''

def gradient_penalty(D, x, x_fake):
    
    # make interpolated sample
    alpha = torch.rand(x.size(0), 1).to(x.device)
    
    alpha = alpha.expand_as(x)  # Expand alpha to match the dimensions of x_real
    #interpolated = alpha * x_real + (1 - alpha) * x_fake
    
    interpolated = alpha * x + (1 - alpha) * x_fake
    interpolated.requires_grad_(True)

    # Forward pass through the critic
    pred = D(interpolated)

    # Compute gradients of the critic’s prediction with respect to the interpolated samples
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones(pred.size(), device=x.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute l2 norm
    gradients = gradients.view(gradients.size(0), -1)  
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)  # small value to avoid sqrt 0

    # Compute the gradient penalty (the penalty is (|grad_norm| - 1)^2)
    gp = torch.mean((grad_norm - 1.0) ** 2)
    
    return gp

def reconstruction_loss(G, x, z):
    G_z = G(z).to(x.device)
    recon_loss = ((G_z - x) ** 2).mean()
    return recon_loss

def D_train_CI_LSGAN(x, G, D, D_optimizer, criterion, lambda_gp):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.rand(x.shape[0], 1) * 0.2 + 0.8
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.rand(x.shape[0], 1).cuda() * 0.2 

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    gp = gradient_penalty(D, x_real, x_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss + lambda_gp * gp
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train_CI_LSGAN(x, G, D, G_optimizer, criterion, lambda_recon):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    recon_loss = reconstruction_loss(G, x, z)

    G_loss_total = G_loss + lambda_recon * recon_loss

    # gradient backprop & optimize ONLY G's parameters
    G_loss_total.backward()
    G_optimizer.step()
        
    return G_loss_total.data.item()


def save_models(G, D, folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(model, folder, filename):
    ckpt_path = os.path.join(folder, filename)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return model

# Function to load images generated at different epochs
def load_images(n_epochs, n_samples, path="samples_exemples"):
    images = []
    for sample in range(n_samples):
        sample_images = []
        for epoch in range(10, n_epochs+1, 10):
            image_path = os.path.join(path, f"epoch_{epoch}_sample_{sample}.png")
            image = Image.open(image_path)
            sample_images.append(image)
        images.append(sample_images)
    return images

# Display generated images to observe evolution
def plot_image_evolution(images, n_epochs, n_samples):
    # Création de la figure
    fig, axes = plt.subplots(nrows=n_epochs // 10, ncols=n_samples, figsize=(n_samples * 2, n_epochs // 5))
    
    # Assure que `axes` est toujours une grille 2D
    if n_epochs // 10 == 1:
        axes = axes[np.newaxis, :]  # Si une seule ligne, ajoute un axe pour le rendre 2D
    if n_samples == 1:
        axes = axes[:, np.newaxis]  # Si une seule colonne, ajoute un axe pour le rendre 2D
    
    # Parcourt et affiche les images dans la grille
    for sample_idx in range(n_samples):
        for epoch_idx, image in enumerate(images[sample_idx]):
            axes[epoch_idx, sample_idx].imshow(transforms.ToTensor()(image).permute(1, 2, 0))
            axes[epoch_idx, sample_idx].axis('off')
            if epoch_idx == 0:
                axes[epoch_idx, sample_idx].set_title(f'Sample {sample_idx + 1}')

    plt.tight_layout()
    plt.show()

