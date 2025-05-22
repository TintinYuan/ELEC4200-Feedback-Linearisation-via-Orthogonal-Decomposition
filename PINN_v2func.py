import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# Import the potential functions from the new module
from PINN_functions import *

# Set random seed for reproducibility
torch.manual_seed(6)
np.random.seed(6)

# Check if GPU is abailable and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Add this line to initialize CUDA context (do get rid of the cuBLAS warning)
if torch.cuda.is_available():
    torch.cuda.init()  # Initialize CUDA
    dummy = torch.zeros(1, device=device)  # Force cuBLAS initialization

class SinusoidalActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# TAG Define the Neural Network for h(x, y, z) ---
class PotentialNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_layers=4, neurons_per_layer=64):
        super(PotentialNet, self).__init__()

        layers = [nn.Linear(input_dim, neurons_per_layer), SinusoidalActivation()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons_per_layer, neurons_per_layer), SinusoidalActivation()])
        layers.append(nn.Linear(neurons_per_layer, output_dim))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        # x is a tensor of shape (N, input_dim)
        return self.sequential(x)

# ATT
# TAG Define the PINN Loss Function ---
# def pinn_loss(model, points, alpha=1.0):
#     """
#     Calculates the PINN loss by comparing the gradient of the model's output
#     to the true vector field at the given points.
#     """
#     # NOTE: Just need one loss function that encaptures the loss ||v_i \nabla h_j = v_j \nabla h_i||
#     # Ensure gradients are computed for the input points
#     points.requires_grad_(True)

#     # Get the model's output h(x, y, z)
#     h_pred = model(points) # Shape: (N, 1)

#     # Compute the gradient of h_pred with respect to the input points (x, y, z)
#     # This is the core of the PINN for gradient matching
#     grad_h_pred = torch.autograd.grad(
#         inputs=points,          # The input tensor w.r.t. which gradients are computed
#         outputs=h_pred,         # The output tensor from which gradients flow
#         grad_outputs=torch.ones_like(h_pred), # Gradient of the output w.r.t. itself (for scalar outputs)
#         create_graph=True,      # Needed to compute higher-order derivatives if necessary (not strictly here, but good practice)
#         retain_graph=True       # Needed if we call autograd.grad or backward multiple times
#     )[0] # autograd.grad returns a tuple, the first element is the gradient tensor. Shape: (N, 3)

#     # Compute the target vector field v(x, y, z) at these points
#     x, y, z = points[:, 0], points[:, 1], points[:, 2]
#     v_target = v_true(x, y, z) # Shape: (N, 3)

#     dim = v_target.shape[1]

#     grad_loss = 0
#     for i in range(dim - 1): # implementation of \|v_i \nabla h_j - v_j \nabla h_i\|
#         proportion_diff = v_target[:, i] * grad_h_pred[:, i + 1] - v_target[:, i + 1] * grad_h_pred[:, i]
#         grad_loss += torch.sum(torch.abs(proportion_diff))

#     return grad_loss

# def proportional_loss_constant(model, points):
#     points.requires_grad_(True)
#     h_pred = model(points)
#     grad_h_pred = torch.autograd.grad(
#         outputs=h_pred,
#         inputs=points,
#         grad_outputs=torch.ones_like(h_pred),
#         create_graph=True,
#         retain_graph=True
#     )[0]

#     x, y, z = points[:, 0], points[:, 1], points[:, 2]
#     v = v_true(x, y, z)

#     numerator = torch.sum(grad_h_pred * v, dim=1)
#     denominator = torch.sum(v * v, dim=1) + 1e-8 # avoid divide by zero
#     c_opt = numerator / denominator

#     v_scaled = c_opt.unsqueeze(1) * v

#     loss = torch.mean((grad_h_pred - v_scaled) ** 2)
#     return loss


def proportional_ratio_loss(model, points, eps=1e-8):
    points.requires_grad_(True)
    h_pred = model(points)
    grad_h = torch.autograd.grad(
        outputs=h_pred,
        inputs=points,
        grad_outputs=torch.ones_like(h_pred),
        create_graph=True,
        retain_graph=True
    )[0]  # shape (N, 3)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    v = v_true(x, y, z)  # shape (N, 3)

    # Avoid division by zero
    v_safe = v + eps * (v.abs() < eps)
    ratio = v_safe / grad_h

    # Loss: squared difference between all component ratios
    diff_01 = (ratio[:, 0] - ratio[:, 1])**2
    diff_02 = (ratio[:, 0] - ratio[:, 2])**2
    diff_12 = (ratio[:, 1] - ratio[:, 2])**2

    # Optional: mask out where any v_i is too small
    valid_mask = (v.abs() > eps).all(dim=1)
    loss = (diff_01 + diff_02 + diff_12)[valid_mask]

    return torch.mean(loss)

# TAG Training Setup ---
model = PotentialNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.89))
epochs = 18000
num_train_points = 10000 # Number of points to sample for training

# Define a domain for sampling points (e.g., a cube)
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
z_min, z_max = -1.0, 1.0

# TAG Training loop ---
loss_history = []

for epoch in range(epochs):
    # Generate random training points within the domain
    # We need to make sure the points tensor can have gradients attached
    train_points = torch.rand(num_train_points, 3, device=device)
    train_points[:, 0] = train_points[:, 0] * (x_max - x_min) + x_min
    train_points[:, 1] = train_points[:, 1] * (y_max - y_min) + y_min
    train_points[:, 2] = train_points[:, 2] * (z_max - z_min) + z_min
    # No need to call .requires_grad_(True) here, done in the loss function for clarity

    # Calculate loss
    loss = proportional_ratio_loss(model, train_points) # NOTE: here the loss is the combined_loss!

    # Backpropagation and optimization
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients of the loss w.r.t. model parameters
    optimizer.step()      # Update model parameters

    # Store and print loss
    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Gradient Loss: {loss.item():.6f}')

print("Training finished.")

# TAG Evaluation ---
# Evaluate the learned function and its gradient on a new set of test points
num_test_points = 1000

test_points = torch.rand(num_test_points, 3, device=device)
test_points[:, 0] = test_points[:, 0] * (x_max - x_min) + x_min
test_points[:, 1] = test_points[:, 1] * (y_max - y_min) + y_min
test_points[:, 2] = test_points[:, 2] * (z_max - z_min) + z_min

# Disable gradient calculation for evaluation
model.eval()

with torch.enable_grad():
    test_points.requires_grad_(True)
    h_pred_test = model(test_points)
    grad_h_pred_test = torch.autograd.grad(
        inputs=test_points,
        outputs=h_pred_test,
        grad_outputs=torch.ones_like(h_pred_test),
        create_graph=False, # No need to create graph for evaluation
        retain_graph=False
    )[0]

# Compute the true values at test points
h_true_test = h_true(test_points[:, 0], test_points[:, 1], test_points[:, 2])
v_true_test = v_true(test_points[:, 0], test_points[:, 1], test_points[:, 2])

# TAG Quantify Accuracy ---

# Accuracy of the learned gradient compared to the true vector field
gradient_error = torch.mean((grad_h_pred_test - v_true_test)**2).item()
print(f"\nMean Squared Error of the Gradient: {gradient_error:.6f}")
print(f"Root Mean Squared Error of the Gradient: {np.sqrt(gradient_error):.6f}")

# Accuracy of the learned potential function compared to the true potential function
# Note: PINNs learn h up to a constant. We can align them by shifting one of them.
# A common way is to match the mean value over the test set.
h_pred_aligned = h_pred_test - torch.mean(h_pred_test) + torch.mean(h_true_test)
potential_error = torch.mean((h_pred_aligned - h_true_test)**2).item()
print(f"Mean Squared Error of the Potential (aligned): {potential_error:.6f}")
print(f"Root Mean Squared Error of the Potential (aligned): {np.sqrt(potential_error):.6f}")

# --- 8. Visualize Results (Optional) ---
# Plotting 3D fields/functions is hard. Let's compare values at a few specific points.
sample_points = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [1.0, -1.0, 0.0],
    [0.0, 1.0, np.pi/2]
], device=device)

model.eval()

with torch.enable_grad():
    sample_points.requires_grad_(True) # Still need grad for v_pred
    h_pred_sample = model(sample_points)
    grad_h_pred_sample = torch.autograd.grad(
        inputs=sample_points,
        outputs=h_pred_sample,
        grad_outputs=torch.ones_like(h_pred_sample),
        create_graph=False, retain_graph=False
    )[0]

h_true_sample = h_true(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2])
v_true_sample = v_true(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2])

print("\n--- Sample Point Comparison ---")
for i in range(sample_points.shape[0]):
    p = sample_points[i].detach().cpu().numpy()
    hp_pred = h_pred_sample[i].detach().cpu().item()
    hp_true = h_true_sample[i].detach().cpu().item()
    vp_pred = grad_h_pred_sample[i].detach().cpu().numpy()
    vp_true = v_true_sample[i].detach().cpu().numpy()

    print(f"Point: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
    print(f"  h_true: {hp_true:.4f}, h_pred: {hp_pred:.4f} (Diff: {abs(hp_true - hp_pred):.4f})")
    print(f"  v_true: [{vp_true[0]:.4f}, {vp_true[1]:.4f}, {vp_true[2]:.4f}]")
    print(f"  v_pred: [{vp_pred[0]:.4f}, {vp_pred[1]:.4f}, {vp_pred[2]:.4f}] (Diff: {np.linalg.norm(vp_true - vp_pred):.4f})")

# Plot training loss history
plt.figure(figsize=(12,6))

log_scale = True

if(log_scale):

    plt.semilogy(loss_history, label='Combined Loss', linewidth=2)
else:

    plt.plot(loss_history, label='Combined Loss', linewidth=2)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss Values (log scale)", fontsize=12)
plt.title("Training Loss History", fontsize=14)

plt.grid(True, which="both", ls='-', alpha=0.2)

plt.legend(fontsize=10)

# Tight layout to ensure everything fits
plt.tight_layout()
plt.show()