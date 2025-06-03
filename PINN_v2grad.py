import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PINN_functions import *

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
loss_fn = CurlFreePKVLoss()

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
    loss = loss_fn(K_output, x_batch, v_batch, grad_v_batch)

    # Backward pass
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss.item()

epochs = 2000
num_train_points = 1000

# Define a domain for sampling points (e.g., a cube)
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
z_min, z_max = -1.0, 1.0

loss_history = []

def generate_training_data(num_train_points=1000):
    """Generate random training samples"""
    coords = torch.rand(num_train_points, 3, device=device)
    coords[:, 0] = coords[:, 0] * (x_max - x_min) + x_min
    coords[:, 1] = coords[:, 1] * (y_max - y_min) + y_min
    coords[:, 2] = coords[:, 2] * (z_max - z_min) + z_min

    return coords

def train_model(num_epochs=2000, batch_size=128):
    model.train()

    for epoch in range(num_epochs):

        # Generate training batch
        x_batch = generate_training_data(batch_size)

        # Training step
        loss_value = train_step(x_batch)
        loss_history.append(loss_value)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss_value:.6f}')

    return model


if __name__ == "__main__":
    # Train the model
    trained_model = train_model(num_epochs=3000, batch_size=128)

    # Test the trained model
    with torch.no_grad():
        test_x = torch.tensor([[1.0, 0.5, 1.0], [0.0, 1.0, 0.5]], requires_grad=True, device=device)
        K_pred = trained_model(test_x)
        print(f"K predictions: {K_pred}")

        # The learned p=exp{K} should make p*v curl-free
        p = torch.exp(K_pred)
        print(f"p = exp(K): {p}")




    plt.figure(figsize=(12, 6))

    plt.semilogy(loss_history, label='Combined Loss', linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss Values (log scale)", fontsize=12)
    plt.title("Training Loss History", fontsize=14)

    plt.grid(True, which="both", ls='-', alpha=0.2)

    plt.tight_layout()
    plt.show()


# def compute_curl(F, coords):
#     grads = []
#     for i in range(3):
#         grad = torch.autograd.grad(F[:, i], coords, grad_outputs=torch.ones_like(F[:, i]), create_graph=True)[0]
#         grads.append(grad)
#     curl_x = grads[1][:, 2] - grads[2][:, 1]
#     curl_y = grads[2][:, 0] - grads[0][:, 2]
#     curl_z = grads[0][:, 1] - grads[1][:, 0]
#     return torch.stack([curl_x, curl_y, curl_z], dim=-1)


# for epoch in range(epochs):
#     coords = torch.rand(num_train_points, 3, device=device) # (x, y, z) in [-1, 1]
#     coords[:, 0] = coords[:, 0] * (x_max - x_min) + x_min
#     coords[:, 1] = coords[:, 1] * (y_max - y_min) + y_min
#     coords[:, 2] = coords[:, 2] * (z_max - z_min) + z_min

#     coords.requires_grad_(True)

#     x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
#     f = model(coords).squeeze()
#     v = vector_v(x, y, z)
#     F = (f.unsqueeze(-1) * v)

#     curl = compute_curl(F, coords)
#     loss = (curl**2).sum(dim=1).mean()

#     loss_history.append(loss.item())

#     optimiser.zero_grad()
#     loss.backward()
#     optimiser.step()

#     if (epoch+1) % 100 == 0:
#         print(f"Step {epoch}, Loss: {loss.item():.6f}")
