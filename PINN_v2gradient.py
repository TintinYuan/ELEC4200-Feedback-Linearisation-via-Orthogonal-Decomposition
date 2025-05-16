import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is abailable and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

class SinusoidalActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# Define neural network for f(x, y, z)
class ScalarFieldF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            SinusoidalActivation(),
            nn.Linear(64, 64),
            SinusoidalActivation(),
            nn.Linear(64, 64),
            SinusoidalActivation(),
            nn.Linear(64, 64),
            SinusoidalActivation(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)  # Output shape: (batch_size, 1)

def vector_v(y, z):
    denom = y**2 + z**2 + 1e-6  # avoid division by zero
    a = 2*z - z*(2*z**2 + y)/denom
    b = -1 + 2*(2*z**2 + y)/denom
    return torch.stack([torch.zeros_like(a), a, b], dim=-1)

def compute_curl(F, coords):
    grads = []
    for i in range(3):
        grad = torch.autograd.grad(F[:, i], coords, grad_outputs=torch.ones_like(F[:, i]), create_graph=True)[0]
        grads.append(grad)
    curl_x = grads[1][:, 2] - grads[2][:, 1]
    curl_y = grads[2][:, 0] - grads[0][:, 2]
    curl_z = grads[0][:, 1] - grads[1][:, 0]
    return torch.stack([curl_x, curl_y, curl_z], dim=-1)

# Sample training loop
model = ScalarFieldF().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 2000
num_train_points = 1000

# Define a domain for sampling points (e.g., a cube)
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
z_min, z_max = -1.0, 1.0

loss_history = []

for epoch in range(epochs):
    coords = torch.rand(num_train_points, 3, device=device) # (x, y, z) in [-1, 1]
    coords[:, 0] = coords[:, 0] * (x_max - x_min) + x_min
    coords[:, 1] = coords[:, 1] * (y_max - y_min) + y_min
    coords[:, 2] = coords[:, 2] * (z_max - z_min) + z_min

    coords.requires_grad_(True)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    f = model(coords).squeeze()
    v = vector_v(y, z)
    F = (f.unsqueeze(-1) * v)

    curl = compute_curl(F, coords)
    loss = (curl**2).sum(dim=1).mean()

    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Step {epoch}, Loss: {loss.item():.6f}")

plt.figure(figsize=(12, 6))

plt.semilogy(loss_history, label='Combined Loss', linewidth=2)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss Values (log scale)", fontsize=12)
plt.title("Training Loss History", fontsize=14)

plt.grid(True, which="both", ls='-', alpha=0.2)

plt.tight_layout()
plt.show()
