import torch 
import torchvision
import os
import argparse


from model import Generator, Discriminator
from utils import load_model
import torch.nn.functional as F

def generate_samples_with_drs(generator, discriminator, batch_size, threshold=0.5):
    accepted_samples = []
    M = 0  # Initialisation de la probabilité maximale

    while len(accepted_samples) < batch_size:
        # Générer un batch complet
        z = torch.randn(batch_size, 100).cuda()
        samples = generator(z)

        # Calculer les scores et probabilités
        scores = discriminator(samples)
        probabilities = F.sigmoid(scores)

        # Mettre à jour M avec la probabilité maximale du batch
        M = max(M, probabilities.max().item())

        # Appliquer la DRS pour chaque échantillon dans le batch
        for i in range(batch_size):
            acceptance_probability = probabilities[i] / M
            if torch.rand(1).item() < acceptance_probability:
                accepted_samples.append(samples[i])
                if len(accepted_samples) >= batch_size:
                    break  # Arrête si le nombre d'échantillons acceptés atteint le batch_size cible

    # Convertir en tenseur et retourner le batch accepté
    return torch.stack(accepted_samples[:batch_size])
'''
def generate_single_sample_with_drs(generator, discriminator, threshold=0.5):
    M = 0  # Initialisation of the maximal probability

    while True:
        z = torch.randn(args.batch_size, 100).cuda()
        sample = generator(z)

        # COmpute the probability
        score = discriminator(sample)
        probability = F.sigmoid(score)

        # New maximal probability
        M = max(M, probability.item())

        acceptance_probability = probability / M  # Probability of rejection
        if torch.rand(1).item() < acceptance_probability:
            return sample  # Return if it is accepted
'''

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

    D = Discriminator(d_input_dim = mnist_dim)
    D = load_model(D, 'checkpoints', 'D.pth')
    D = torch.nn.DataParallel(D).cuda()
    D.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            
            x = generate_samples_with_drs(G, D, batch_size=args.batch_size, threshold = 0.5)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1


    
