import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(88)
np.random.seed(88)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# --- 1. Define the true scalar function and its gradient (the vector field) ---
# We'll use this only to generate the target vector field v and potentially for validation
def h_true(x, y, z):
    """The true scalar potential function."""
    return x**2 + y**3 + torch.sin(z)

def h_true2(x, y, z):
    c1 = 2.0
    c2 = 1.0
    c = 3.0
    theta = 1.44
    k = 1.0
    return c1*(theta*y**2 + c*z**2) + c2

def v_true(x, y, z):
    """The true conservative vector field (gradient of h_true)."""
    # Compute partial derivatives symbolically or manually
    dv_dx = 2 * x
    dv_dy = 3 * y**2
    dv_dz = torch.cos(z)
    return torch.stack([dv_dx, dv_dy, dv_dz], dim=-1)

def v_true2(x, y, z):
    c1 = 2.0
    c2 = 1.0
    c = 3.0
    theta = 1.44
    k = 1.0

    dv_dx = 0
    dv_dy = c1 * 2 * theta * y
    dv_dz = c1 * 2 * c * z
    return torch.stack([dv_dx, dv_dy, dv_dz], dim=-1)

# --- 2. Define the Neural Network for h(x, y, z) ---
class PotentialNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_layers=4, neurons_per_layer=50):
        super(PotentialNet, self).__init__()

        layers = [nn.Linear(input_dim, neurons_per_layer), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons_per_layer, neurons_per_layer), nn.Tanh()])
        layers.append(nn.Linear(neurons_per_layer, output_dim))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        # x is a tensor of shape (N, input_dim)
        return self.sequential(x)

# --- 3. Define the PINN Loss Function ---
# TAG Loss function
def pinn_loss(model, points, alpha=1.0):
    """
    Calculates the PINN loss by comparing the gradient of the model's output
    to the true vector field at the given points.
    """
    # NOTE: Just need one loss function that encaptures the loss ||v_i \nabla h_j = v_j \nabla h_i||
    # Ensure gradients are computed for the input points
    points.requires_grad_(True)

    # Get the model's output h(x, y, z)
    h_pred = model(points) # Shape: (N, 1)

    # Compute the gradient of h_pred with respect to the input points (x, y, z)
    # This is the core of the PINN for gradient matching
    grad_h_pred = torch.autograd.grad(
        inputs=points,          # The input tensor w.r.t. which gradients are computed
        outputs=h_pred,         # The output tensor from which gradients flow
        grad_outputs=torch.ones_like(h_pred), # Gradient of the output w.r.t. itself (for scalar outputs)
        create_graph=True,      # Needed to compute higher-order derivatives if necessary (not strictly here, but good practice)
        retain_graph=True       # Needed if we call autograd.grad or backward multiple times
    )[0] # autograd.grad returns a tuple, the first element is the gradient tensor. Shape: (N, 3)

    # Compute the target vector field v(x, y, z) at these points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    v_target = v_true(x, y, z) # Shape: (N, 3)
    h_target = h_true(x, y, z).unsqueeze(1)

    # Calculate the mean squared error between the predicted gradient and the target vector field
    gradient_loss = torch.mean((grad_h_pred - v_target)**2)
    function_loss = torch.mean((h_pred - h_target)**2)

    # Combine losses using the wiehgt factor alone:
    combined_loss = alpha * gradient_loss + (1 - alpha)*function_loss

    return combined_loss, gradient_loss, function_loss

# --- 4. Training Setup ---
model = PotentialNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.89)) # TODO changed
epochs = 10000
num_train_points = 10000 # Number of points to sample for training

# Define a domain for sampling points (e.g., a cube)
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
z_min, z_max = -1.0, 1.0

# --- 5. Training Loop ---
# TAG Training loop
loss_history = []
gradient_loss_history = []
function_loss_history = []

for epoch in range(epochs):
    # Generate random training points within the domain
    # We need to make sure the points tensor can have gradients attached
    train_points = torch.rand(num_train_points, 3, device=device)
    train_points[:, 0] = train_points[:, 0] * (x_max - x_min) + x_min
    train_points[:, 1] = train_points[:, 1] * (y_max - y_min) + y_min
    train_points[:, 2] = train_points[:, 2] * (z_max - z_min) + z_min
    # No need to call .requires_grad_(True) here, done in the loss function for clarity

    # Calculate loss
    loss, grad_loss, func_loss = pinn_loss(model, train_points, alpha=1.0) # NOTE: here the loss is the combined_loss!

    # Backpropagation and optimization
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients of the loss w.r.t. model parameters
    optimizer.step()      # Update model parameters

    # Store and print loss
    loss_history.append(loss.item())
    gradient_loss_history.append(grad_loss.item())
    function_loss_history.append(func_loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Combined Loss: {loss.item():.6f},'
              f' Gradient Loss: {grad_loss.item():.6f}, Function loss: {func_loss.item():.6f}')

print("Training finished.")

# --- 6. Evaluation ---
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

# --- 7. Quantify Accuracy ---

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
    print(f"  h_true: {hp_true:.4f}")
    print(f"  h_pred: {hp_pred:.4f} (Diff: {abs(hp_true - hp_pred):.4f})")
    print(f"  v_true: [{vp_true[0]:.4f}, {vp_true[1]:.4f}, {vp_true[2]:.4f}]")
    print(f"  v_pred: [{vp_pred[0]:.4f}, {vp_pred[1]:.4f}, {vp_pred[2]:.4f}] (Diff: {np.linalg.norm(vp_true - vp_pred):.4f})")

# Plot training loss history
plt.figure(figsize=(12,6))

log_scale = True

if(log_scale):

    plt.semilogy(loss_history, label='Combined Loss', linewidth=2)
    plt.semilogy(gradient_loss_history, label='Gradient Loss', linewidth=2)
    plt.semilogy(function_loss_history, label='Function Loss', linewidth=2)
else:

    plt.plot(loss_history, label='Combined Loss', linewidth=2)
    plt.plot(gradient_loss_history, label='Gradient Loss', linewidth=2)
    plt.plot(function_loss_history, label='Function Loss', linewidth=2)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss Values (log scale)", fontsize=12)
plt.title("Training Loss History", fontsize=14)

plt.grid(True, which="both", ls='-', alpha=0.2)

plt.legend(fontsize=10)

# Tight layout to ensure everything fits
plt.tight_layout()
plt.show()