import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PINNs.PINN_v2func import *

# For a nonlinear system: ẋ = f(x) + g(x)u
# Feedback linearization seeks: u = α(x) + β(x)v 
# Such that closed-loop system becomes linear: ẏ = v

class FeedbackLinearisationPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dim = 3
        self.control_dim = 2
        # Network for a(x)
        self.alpha_net = PotentialNet(input_dim=self.state_dim, output_dim=self.control_dim)
        # Network for β(x)
        self.beta_net = PotentialNet(input_dim=self.state_dim, output_dim=self.control_dim*self.state_dim)
        
    def forward(self, x):
        alpha = self.alpha_net(x)
        beta = self.beta_net(x).reshape(-1, self.control_dim, self.state_dim)
        return alpha, beta
        
    def control_law(self, x, v):
        alpha, beta = self.forward(x)
        # u = α(x) + β(x)v
        u = alpha + torch.bmm(beta, v.unsqueeze(-1)).squeeze(-1)
        return u
    
def compute_linearisation_error(x_batch, alpha, beta, f_x, g_x):
    
    return
    

def feedback_linearization_loss(model, x_batch, system_model):
    alpha, beta = model(x_batch)
    
    # Get system dynamics (could be learned separately or known)
    f_x, g_x = system_model(x_batch)
    
    # Compute Lie derivatives (for relative degree verification)
    # ...
    
    # Compute linearization error (closed-loop system should be linear)
    linear_error = compute_linearisation_error(x_batch, alpha, beta, f_x, g_x)
    
    # Add regularization for zero dynamics stability if needed
    # ...
    
    return linear_error
