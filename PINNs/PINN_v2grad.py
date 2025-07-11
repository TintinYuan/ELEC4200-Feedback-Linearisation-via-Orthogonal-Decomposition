import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PINNs.PINN_functions import *

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is abailable and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

class SinusoidalActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# Define neural network that outputs K(x)
class KNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1, activation_fn=nn.Tanh()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)  # Output shape: (batch_size, 1)

def vector_v(x_single_torch):
    x = x_single_torch[0]
    y = x_single_torch[1]
    z = x_single_torch[2]

    denom = y**2 + z**2 + 1e-6  # avoid division by zero
    a = 2*z - z*(2*z**2 + y)/denom
    b = -1 + 2*(2*z**2 + y)/denom
    return torch.stack([torch.zeros_like(a), a, b], dim=-1)


# Sample training loop
model = KNetwork().to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Specify curl-free loss
curl_loss_fn = v2grad_CurlFreepvLoss()

# Specify point constraint loss
point_constraint_loss_fn = v2grad_PointConstraintLoss(
    constraint_points=[[1.0, 1.0, 1.0]],
    constraint_values=[1.0]
)

# Point constraint weight
constraint_weight = 10.0

# TODO Train step
def train_step(x_batch):
    """Single training step"""

    # Prepare input with gradient tracking
    x_batch = x_batch.to(device).requires_grad_(True)
    batch_size = x_batch.shape[0]

    # Forward path through K network
    K_output = model(x_batch)

    # Compute vector field v(x) for the batch
    v_batch = v_pytorch_batch(vector_v, x_batch)

    # Compute Jacobian of v w.r.t. x
    x_for_jacobian = x_batch.detach().clone().requires_grad_(True)
    grad_v_batch = v_jacobian(vector_v, x_for_jacobian)

    # Compute loss
    curl_loss = curl_loss_fn(K_output, x_batch, v_batch, grad_v_batch)
    constraint_loss = point_constraint_loss_fn(model)
    total_loss = curl_loss + constraint_weight * constraint_loss

    # Backward pass
    optimiser.zero_grad()
    total_loss.backward()
    optimiser.step()

    return total_loss.item(), curl_loss.item(), constraint_weight * constraint_loss.item()

epochs = 2000
num_train_points = 1000

# Define a domain for sampling points (e.g., a cube)
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
z_min, z_max = -1.0, 1.0
domain = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

def train_model(num_epochs=2000, batch_size=128):
    model.train()

    total_loss_history = []
    curl_loss_history = []
    constraint_loss_history = []

    for epoch in range(num_epochs):

        # Generate training batch
        x_batch = generate_training_data(domain, batch_size)

        # Training step
        total_loss_value, curl_loss_value, constraint_loss_value = train_step(x_batch)
        
        # Store all loss components
        total_loss_history.append(total_loss_value)
        curl_loss_history.append(curl_loss_value)
        constraint_loss_history.append(constraint_loss_value)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Total Loss: {total_loss_value:.3f}, '
                  f'Curl Loss: {curl_loss_value:.3f}, Constraint Loss: {constraint_loss_value:.3f}')

    return model, total_loss_history, curl_loss_history, constraint_loss_history


if __name__ == "__main__":
    # Train the model
    trained_model, total_losses, curl_losses, constraint_losses = train_model(num_epochs=3000, batch_size=128)

    # Test the trained model
    with torch.no_grad():
        test_x = torch.tensor([[1.0, 0.5, 1.0], [0.0, 1.0, 0.5]], requires_grad=True, device=device)
        K_pred = trained_model(test_x)
        print(f"K predictions: {K_pred}")

        # The learned p=exp{K} should make p*v curl-free
        p = torch.exp(K_pred)
        print(f"p = exp(K): {p}")




    plt.figure(figsize=(12, 6))

    plt.semilogy(total_losses, label='Total loss', linewidth=2, color='blue')
    plt.semilogy(curl_losses, label='Curl loss', linewidth=2, color='red')
    plt.semilogy(constraint_losses, label='Constraint loss', linewidth=2, color='green')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss Values (log scale)", fontsize=12)
    plt.title("Training Loss History", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls='-', alpha=0.2)

    plt.tight_layout()
    plt.show()

