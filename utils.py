import torch
import os
from torch.distributions import MultivariateNormal, Categorical

def sample_gmm(batch_size, n_components=10, latent_dim=100, c=1.0, sigma=0.1):
    """
    Args:
    # n_components = K, number of Gaussians in the mixture (10 for MNIST cuz 10 digit types)
    # latent_dim = d, dimension of the latent space (should match Generator's input)
    # c = c, range of the uniform distribution, hyperparameter, [-c, c] (default: 1.0)
    # sigma = sigma, standard deviation of the covariance matrices, hyperparameter (default: 0.1)
    # batch_size = N, number of samples to generate

    Returns:
        torch.Tensor: Samples from GMM of shape (batch_size, latent_dim)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    means = torch.randn(n_components, latent_dim).to(device) * c # fixed means for components 
    
    cov = torch.eye(latent_dim).to(device) * (sigma ** 2) # covariance matrix (sigma * I) for all components
    
    # Randomly select components for each sample in the batch
    selected_components = torch.randint(0, n_components, (batch_size,)).to(device)
    
    # Generate samples
    samples = torch.zeros(batch_size, latent_dim).to(device)
    
    # Sample from selected components
    for k in range(n_components):
        mask = (selected_components == k)
        if mask.sum() > 0:
            dist = MultivariateNormal(means[k], cov)
            samples[mask] = dist.sample((mask.sum(),))
    
    # Normalize samples to have similar scale as standard normal
    samples = samples / (latent_dim ** 0.5)
    
    return samples

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()
    
    # train discriminator on real
    # x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x, torch.rand(x.shape[0], 1) * 0.2 + 0.8 # Yannis' modification
    x_real, y_real = x_real.cuda(), y_real.cuda()
    
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output
    
    # train discriminator on fake
    # Use GMM instead of standard normal distribution
    z = sample_gmm(batch_size=x.shape[0])
    # x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()
    x_fake, y_fake = G(z), torch.rand(x.shape[0], 1).cuda() * 0.2 # Yannis' modification
    # print("x_fake", x_fake.shape)
    # print("y_fake", y_fake.shape)
    D_output = D(x_fake)
    # print("D_output", D_output.shape)
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
    z = sample_gmm(batch_size=x.shape[0])
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
