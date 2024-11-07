import torch 
import os
from tqdm import trange
import argparse
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



from model import Generator, Discriminator
from utils import D_train_CI_LSGAN, G_train_CI_LSGAN, save_models, load_images, plot_image_evolution

def gradient_penalty(D, x):
    x.requires_grad_(True)
    d_out = D(x)
    grad_x = torch.autograd.grad(outputs=d_out, inputs=x,
                                 grad_outputs=torch.ones_like(d_out),
                                 create_graph=True)[0]
    grad_penalty = ((grad_x.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def reconstruction_loss(G, x, z):
    G_z = G(z)
    recon_loss = ((G_z - x) ** 2).mean()
    return recon_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.MSELoss()

    lambda_gp = 0.01  # Weight for gradient penalty
    lambda_recon = 0.001  # Weight for reconstruction

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    print('Start Training :')
    D_losses = []
    G_losses = []
    epochs = []
    z_exemple = torch.randn(6, 100).cuda()  # Génère un batch de bruit
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)

            D_loss = D_train_CI_LSGAN(x, G, D, D_optimizer, criterion, lambda_gp)
            G_loss = G_train_CI_LSGAN(x, G, D, G_optimizer, criterion, lambda_recon)

        D_losses.append(D_loss) 
        G_losses.append(G_loss)
        epochs.append(epoch)

        if epoch % 10 == 0 or epoch == 1:
            n_samples = 0 
            save_models(G, D, 'checkpoints')
            with torch.no_grad(): 
                generated_images = G(z_exemple)
                generated_images = generated_images.reshape(-1, 28, 28)
                
                for k in range(generated_images.size(0)):
                    image_path = os.path.join("samples_exemples", f"epoch_{epoch}_sample_{n_samples}.png")
                    torchvision.utils.save_image(generated_images[k:k+1], image_path)
                    n_samples += 1  # Incrémente le compteur d'images
            
                
    print('Training done')

plt.figure(figsize=(10, 5))
plt.plot(D_losses, label="Discriminator Loss")
plt.plot(G_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss of Discriminator and Generator over Epochs")
plt.legend()
plt.show()

images = load_images(n_epoch, n_samples)
plot_image_evolution(images, n_epoch, n_samples)
