import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
# Import the potential functions from the new module
from PINN_functions import *

# Set random seed for reproducibility
torch.manual_seed(16)
np.random.seed(16)

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

        layers = []

        # First layer
        first_layer = nn.Linear(input_dim, neurons_per_layer)
        # # Xavier initialisation for sinusoidal activation
        # nn.init.xavier_uniform_(first_layer.weight)
        # nn.init.zeros_(first_layer.bias)
        layers.extend([first_layer, nn.Tanh()])

        # Hidden layers
        for _ in range(hidden_layers - 1):
            hidden_layer = nn.Linear(neurons_per_layer, neurons_per_layer)
            # nn.init.xavier_uniform_(hidden_layer.weight)
            # nn.init.zeros_(hidden_layer.bias)
            layers.extend([hidden_layer, nn.Tanh()])

        # Output layer
        output_layer = nn.Linear(neurons_per_layer, output_dim)
        # nn.init.xavier_uniform_(output_layer.weight)
        # nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.sequential = nn.Sequential(*layers) # here * is a unpacking argument

    def forward(self, x):
        # x is a tensor of shape (N, input_dim)
        return self.sequential(x)

# TAG Training Setup ---
model = PotentialNet().to(device)

# the following line to initialise CUDA to avoid warning
# Initialize CUDA context specifically for gradient computations
if torch.cuda.is_available():
    # Create a dummy computation that uses the same gradient operations
    dummy_input = torch.randn(1, 3, device=device, requires_grad=True)  # Adjust shape as needed
    dummy_output = dummy_input.sum()
    
    # This will initialize cuBLAS for gradient computations
    dummy_grad = torch.autograd.grad(
        outputs=dummy_output,
        inputs=dummy_input,
        grad_outputs=torch.ones_like(dummy_output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Clean up
    del dummy_input, dummy_output, dummy_grad
    torch.cuda.empty_cache()

optimiser = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=500, factor=0.5)

# Define loss function through classes
cross_loss_fn = v2func_cross_loss()
zero_grad_loss_fn = v2func_zero_grad_loss()

epochs = 3000
num_train_points = 10000 # Number of points to sample for training
point_range = 1
# Define a domain for sampling points (e.g., a cube)
x_min, x_max = -point_range, point_range
y_min, y_max = -point_range, point_range
z_min, z_max = -point_range, point_range
domain = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

# TAG Training loop ---
total_loss_history = []
cross_loss_history = []
zero_grad_loss_history = []

best_loss = float('inf')
patience_counter = 0
early_stop_patience = 2000


print("Starting training...")

for epoch in range(epochs):
    model.train()

    # Generate random training points within the domain
    x_batch = generate_training_data(domain, num_train_points)
    x_batch = x_batch.to(device).requires_grad_(True)
    v_batch = v_pytorch_batch(v_true2_tensor, x_batch)
    # print(f"Model device: {next(model.parameters()).device}")
    # print(f"Data device: {train_points.device}")
    K_output = model(x_batch)

    # TAG Calculate loss

    cross_loss_val = cross_loss_fn(K_output, x_batch, v_batch)
    zero_grad_loss_val = zero_grad_loss_fn(K_output, x_batch)

    alpha = 0.8
    loss = alpha*cross_loss_val + (1 - alpha)*zero_grad_loss_val

    # Backpropagation and optimization
    optimiser.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients of the loss w.r.t. model parameters

    # HACK
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimiser.step()      # Update model parameters
    scheduler.step(loss)

    # Store and print loss
    total_loss_val = loss.item()
    total_loss_history.append(total_loss_val)

    cross_loss_val = cross_loss_val.item()
    cross_loss_history.append(cross_loss_val)

    zero_grad_loss_val = zero_grad_loss_val.item()
    zero_grad_loss_history.append(zero_grad_loss_val)

    # Early stopping check
    if total_loss_val < best_loss:
        best_loss = total_loss_val
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        break

    if (epoch + 1) % 10 == 0:
        current_lr = optimiser.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{epochs}], Total loss: {total_loss_val:.8f}, LR: {current_lr:.2e}')

print("Training finished.")

# SUPTAG Evaluation ---
# Evaluate the learned function and its gradient on a new set of test points
model.eval() # Disable gradient calculation for evaluation
num_test_points = 2000

test_points = torch.rand(num_test_points, 3, device=device)
test_points[:, 0] = test_points[:, 0] * (x_max - x_min) + x_min
test_points[:, 1] = test_points[:, 1] * (y_max - y_min) + y_min
test_points[:, 2] = test_points[:, 2] * (z_max - z_min) + z_min

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
v_true_test = v_true2(test_points[:, 0], test_points[:, 1], test_points[:, 2])

cross_error, cosine_error = compute_proportionality_metrics(grad_h_pred_test, v_true_test)

# SUPTAG Value printing
print(f"\n--- Enhanced Accuracy Metrics ---")
print(f"Mean Cross Product Error: {cross_error:.8f}")
print(f"Mean Cosine Similarity Error: {cosine_error:.8f}")

n_points = grad_h_pred_test.shape[0]
random_indices = torch.randperm(n_points)[:10]

# Safe data and figures
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_fig_dir = os.path.join('figures')
output_data_dir = os.path.join('data')
os.makedirs(output_fig_dir, exist_ok=True)
os.makedirs(output_data_dir, exist_ok=True)

safe_filename = os.path.join(output_data_dir, f"gradient_comparison_{current_time}.txt")

with open(safe_filename, 'w') as f:
    f.write("Raw values at 10 randomly selected points:\n")
    f.write("=" * 60 + "\n\n")

    print("Raw values at 10 ramdonly selected points")
    print("=" * 60)

    for i, idx in enumerate(random_indices):
        grad_pred = grad_h_pred_test[idx].detach().cpu().numpy()
        v_true = v_true_test[idx].detach().cpu().numpy()

        # Format the output strings
        point_info = f"Point {i+1} (index {idx.item()}):"
        grad_line = f"  grad_h_pred: [{grad_pred[0]:8.5f}, {grad_pred[1]:8.5f}, {grad_pred[2]:8.5f}]"
        true_line = f"  v_true: [{v_true[0]:8.5f}, {v_true[1]:8.5f}, {v_true[2]:8.5f}]"
        proportion_line = f"  dh/v: [{grad_pred[0]/v_true[0]:8.5f}, {grad_pred[1]/v_true[1]:8.5f}, {grad_pred[2]/v_true[2]:8.5f}]"

        # Print to console
        print(point_info)
        print(grad_line)
        print(true_line)
        print(proportion_line)
        print()

        # Write to file
        f.write(point_info + "\n")
        f.write(grad_line + "\n")
        f.write(true_line + "\n")
        f.write(proportion_line + "\n\n")

# gradient_error = torch.mean((grad_h_pred_test - v_true_test)**2).item()
# print(f"Mean Squared Error of Gradient: {gradient_error:.8f}")

# Plot results with better visualisation
plt.figure(figsize=(15, 5))

# Loss history
plt.subplot(1, 3, 1)
if len(total_loss_history) > 100:
    # Show both full history and recent history
    plt.semilogy(total_loss_history, alpha=0.7, linewidth=1, color='blue', label='Total loss')
    # plt.semilogy(total_loss_history[-1000:], linewidth=2, color='red', label='Recent')
    plt.semilogy(cross_loss_history, linewidth=1, color='red', label='Cross loss')
    plt.semilogy(zero_grad_loss_history, linewidth=1, color='green', label='Zero-grad loss')
    plt.legend()
else:
    plt.semilogy(total_loss_history, linewidth=2)

plt.xlabel("Eproch")
plt.ylabel("Loss (log scale)")
plt.title("Training loss history")
plt.grid(True, alpha=0.3)


# Gradient comparison at sample points
plt.subplot(1, 3, 2)
sample_indices = torch.randperm(min(100, num_test_points))[:50]
grad_magnitude_pred = torch.norm(grad_h_pred_test[sample_indices], dim=1).detach().cpu()
grad_magnitude_true = torch.norm(v_true_test[sample_indices], dim=1).detach().cpu()

plt.scatter(grad_magnitude_true, grad_magnitude_pred, alpha=0.6)
max_val = max(grad_magnitude_true.max(), grad_magnitude_pred.max())
plt.plot([0, max_val], [0, max_val], 'r--', label='Prefect match')
plt.xlabel("True gradient magnitude")
plt.ylabel("Predicted gradient magnitude")
plt.title("Gradient magnitude comparison")
plt.legend()
plt.grid(True, alpha=0.3)

# Cross product error distribution
plt.subplot(1, 3, 3)
cross_prod = torch.cross(grad_h_pred_test, v_true_test, dim=1)
cross_magnitude = torch.norm(cross_prod, dim=1).detach().cpu()
plt.hist(cross_magnitude.numpy(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel("Cross product magnitude")
plt.ylabel("Frequency")
plt.title("Cross product error distribution")
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()

safe_filename = os.path.join(output_fig_dir, f"loss_plot_{current_time}.jpg")
plt.savefig(safe_filename, format='jpeg', dpi=600)

plt.show()

# region
# TAG Quantify Accuracy ---

# # Accuracy of the learned gradient compared to the true vector field
# gradient_error = torch.mean((grad_h_pred_test - v_true_test)**2).item()
# print(f"\nMean Squared Error of the Gradient: {gradient_error:.6f}")
# print(f"Root Mean Squared Error of the Gradient: {np.sqrt(gradient_error):.6f}")

# # Accuracy of the learned potential function compared to the true potential function
# # Note: PINNs learn h up to a constant. We can align them by shifting one of them.
# # A common way is to match the mean value over the test set.
# h_pred_aligned = h_pred_test - torch.mean(h_pred_test) + torch.mean(h_true_test)
# potential_error = torch.mean((h_pred_aligned - h_true_test)**2).item()
# print(f"Mean Squared Error of the Potential (aligned): {potential_error:.6f}")
# print(f"Root Mean Squared Error of the Potential (aligned): {np.sqrt(potential_error):.6f}")

# --- 8. Visualize Results (Optional) ---
# # Plotting 3D fields/functions is hard. Let's compare values at a few specific points.
# sample_points = torch.tensor([
#     [0.0, 0.0, 0.0],
#     [0.5, 0.5, 0.5],
#     [-0.5, -0.5, -0.5],
#     [1.0, -1.0, 0.0],
#     [0.0, 1.0, np.pi/2]
# ], device=device)

# model.eval()

# with torch.enable_grad():
#     sample_points.requires_grad_(True) # Still need grad for v_pred
#     h_pred_sample = model(sample_points)
#     grad_h_pred_sample = torch.autograd.grad(
#         inputs=sample_points,
#         outputs=h_pred_sample,
#         grad_outputs=torch.ones_like(h_pred_sample),
#         create_graph=False, retain_graph=False
#     )[0]

# h_true_sample = h_true(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2])
# v_true_sample = v_true(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2])

# print("\n--- Sample Point Comparison ---")
# for i in range(sample_points.shape[0]):
#     p = sample_points[i].detach().cpu().numpy()
#     hp_pred = h_pred_sample[i].detach().cpu().item()
#     hp_true = h_true_sample[i].detach().cpu().item()
#     vp_pred = grad_h_pred_sample[i].detach().cpu().numpy()
#     vp_true = v_true_sample[i].detach().cpu().numpy()

#     print(f"Point: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
#     print(f"  h_true: {hp_true:.4f}, h_pred: {hp_pred:.4f} (Diff: {abs(hp_true - hp_pred):.4f})")
#     print(f"  v_true: [{vp_true[0]:.4f}, {vp_true[1]:.4f}, {vp_true[2]:.4f}]")
#     print(f"  v_pred: [{vp_pred[0]:.4f}, {vp_pred[1]:.4f}, {vp_pred[2]:.4f}] (Diff: {np.linalg.norm(vp_true - vp_pred):.4f})")

# # Plot training loss history
# plt.figure(figsize=(12,6))

# log_scale = True

# if(log_scale):

#     plt.semilogy(loss_history, label='Combined Loss', linewidth=2)
# else:

#     plt.plot(loss_history, label='Combined Loss', linewidth=2)

# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Loss Values (log scale)", fontsize=12)
# plt.title("Training Loss History", fontsize=14)

# plt.grid(True, which="both", ls='-', alpha=0.2)

# plt.legend(fontsize=10)

# # Tight layout to ensure everything fits
# plt.tight_layout()
# plt.show()
# endregion