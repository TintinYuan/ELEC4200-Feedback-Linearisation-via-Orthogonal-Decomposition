import torch
import torch.nn as nn
import torch.autograd as autograd

# === 1. Define the vector field v(x) ===
# Example: non-conservative field
def vector_field(x):
    # x: [batch_size, 3]
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    v1 = -x2
    v2 = x1
    v3 = torch.zeros_like(x1)
    return torch.stack([v1, v2, v3], dim=1)  # [batch_size, 3]

# === 2. Neural network for integrating factor p(x) ===
class IntegratingFactorNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [batch_size]

# === 3. Compute curl of p(x)*v(x) ===
def curl_pv(x, model):
    x.requires_grad_(True)
    p = model(x)  # [batch_size]
    v = vector_field(x)  # [batch_size, 3]
    pv = p.unsqueeze(1) * v  # [batch_size, 3]

    # Compute partial derivatives using autograd
    grads = []
    for i in range(3):
        grad = autograd.grad(pv[:, i], x, grad_outputs=torch.ones_like(pv[:, i]),
                             create_graph=True)[0]
        grads.append(grad)  # Each grad: [batch_size, 3]

    # Compute curl components
    curl_x = grads[2][:, 1] - grads[1][:, 2]
    curl_y = grads[0][:, 2] - grads[2][:, 0]
    curl_z = grads[1][:, 0] - grads[0][:, 1]

    curl = torch.stack([curl_x, curl_y, curl_z], dim=1)  # [batch_size, 3]
    return curl

# === 4. Training setup ===
model = IntegratingFactorNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10000):
    x = torch.rand(256, 3) * 2 - 1  # sample from [-1, 1]^3
    curl = curl_pv(x, model)
    loss = (curl ** 2).sum(dim=1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
