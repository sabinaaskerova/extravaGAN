import torch
import os
from torch.autograd import Variable
from torch.autograd import grad as torch_grad



def D_train(x, G, D, D_optimizer, criterion, gp_weight=10):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    

    # train discriminator on facke
    # z = torch.randn(x.shape[0], 100).cuda()
    # x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()
    # z = torch.randn(x.shape[0], 100)
    # x_fake = G(z)
    # y_fake =  D(x_fake)

    # x_real = x
    # y_real = D(x_real)

    # D_cost = criterion(y_real, y_fake)

    # gp = gradient_penalty(D, x_real, x_fake)

    
    
    # # gradient backprop & optimize ONLY D's parameters
    # D_loss = D_cost + gp * gp_weight
    # D_loss.backward()
    # D_optimizer.step()

    z = torch.randn(x.shape[0], 100)
    x_fake = G(z)
    y_real = D(x)
    y_fake = D(x_fake)
    #loss_D = criterion(y_real, y_fake, gp_weight, gradient_penalty(D, x, x_fake))
    loss_D = criterion(y_real, y_fake, gp_weight, gradient_penalty(D, x, x_fake))

    loss_D.backward()  # Ensure you call backward to compute gradients
    # for name, param in D.named_parameters():
    #     if param.grad is not None:
    #         print(f"Layer {name} | Grad Norm: {param.grad.norm()}")
    #     else:
    #         print(f"Layer {name} | No gradient")
    D_optimizer.step()
    #print(loss_D)
        
    return  loss_D.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    

    # z = torch.randn(x.shape[0], 100).cuda()
    # y = torch.ones(x.shape[0], 1).cuda()
    G.zero_grad()
    z = torch.randn(x.shape[0], 100)
    y = torch.ones(x.shape[0], 1)
    
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output)

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

def gradient_penalty(D, x, x_fake):
    
    # make interpolated sample
    alpha = torch.rand(x.size(0), 1)
    
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
    D.zero_grad()
    
    return gp

# def gradient_penalty(D, real_data, generated_data):
#     D.zero_grad()
#     batch_size = real_data.size()[0]

#     # Calculate interpolation
#     alpha = torch.rand(batch_size, 1, 1, 1)
#     #alpha = alpha.expand_as(real_data)
    
#     interpolated = alpha * real_data + (1 - alpha) * generated_data
#     interpolated = Variable(interpolated, requires_grad=True)
    

#     # Calculate probability of interpolated examples
#     prob_interpolated = D(interpolated)

#     # Calculate gradients of probabilities with respect to examples
#     gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
#                             grad_outputs=torch.ones(prob_interpolated.size()),
#                             create_graph=True, retain_graph=True)[0]

#     # Gradients have shape (batch_size, num_channels, img_width, img_height),
#     # so flatten to easily take norm per example in batch
#     gradients = gradients.view(batch_size, -1)
#     #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

#     # Derivatives of the gradient close to 0 can cause problems because of
#     # the square root, so manually calculate norm and add epsilon
#     gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
#     D.zero_grad()
#     # Return gradient penalty
#     return ((gradients_norm - 1) ** 2).mean()


def D_loss(y_real, y_fake, gp_weight, gp):
    #this approximates the Wasserstein distance to calculate the loss
    return torch.mean(y_fake) - torch.mean(y_real) + gp_weight * gp
def G_loss(y_fake):
    return -torch.mean(y_fake)