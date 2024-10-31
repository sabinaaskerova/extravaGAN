import torch
import numpy as np
import os
from torch.distributions.multivariate_normal import MultivariateNormal

class StaticGMM:
    def __init__(self, n_components=10, latent_dim=100, c=1.0, sigma=0.1):
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.c = c
        self.sigma = sigma

        # Initialize the means of each component uniformly in U[-c, c]^d
        self.means = np.random.uniform(-c, c, (n_components, latent_dim))

        # Initialize full covariance matrices for each component
        self.covariances = np.array([sigma * np.eye(latent_dim) for _ in range(n_components)])

        # Convert means and covariances to PyTorch tensors for efficiency
        self.means = torch.tensor(self.means, dtype=torch.float32)
        self.covariances = torch.tensor(self.covariances, dtype=torch.float32)

    def sample(self, batch_size):
        # Generate random indices for the components
        component_indices = np.random.choice(self.n_components, size=batch_size)
        
        # Initialize empty tensor to store samples
        samples = torch.zeros((batch_size, self.latent_dim))

        for i in range(batch_size):
            # Get the mean and covariance for the selected component
            mean = self.means[component_indices[i]]
            covariance = self.covariances[component_indices[i]]
            
            # Sample from the multivariate normal distribution for this component
            mvn = MultivariateNormal(mean, covariance_matrix=covariance)
            samples[i] = mvn.sample()
        
        return samples
    
static_gmm = StaticGMM(n_components=10, latent_dim=100, c=0.1, sigma=0.15)

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
    # Use Static GMM instead of standard normal distribution
    z = static_gmm.sample(batch_size=x.shape[0])
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

    # Use Static GMM instead of standard normal distribution
    z = static_gmm.sample(batch_size=x.shape[0])
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
