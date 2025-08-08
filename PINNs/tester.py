import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(88)
np.random.seed(88)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Define a neural network to represent the scalar potential function
class PotentialNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Assuming 3D space (x,y,z)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Scalar output
        )
    
    def forward(self, x):
        return self.net(x)
    
    def gradient(self, x):
        """Compute the gradient of the scalar function"""
        x.requires_grad_(True)
        y = self.forward(x)
        
        # Create gradient vector
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        
        return grad

# Convert sympy vector field to numpy/torch
def convert_sympy_to_torch(sympy_vector_field, points, device="cpu"):
    """
    Convert a sympy vector field to torch tensors evaluated at given points
    """
    # Define sympy variables
    x, y, z = sp.symbols('x y z')
    variables = [x, y, z]
    
    # Convert sympy expression to numpy function
    vector_funcs = []
    for component in sympy_vector_field:
        vector_funcs.append(sp.lambdify(variables, component, 'numpy'))
    
    # Evaluate at points
    vector_field_values = []
    for i in range(len(vector_funcs)):
        vector_field_values.append(vector_funcs[i](points[:, 0], points[:, 1], points[:, 2]))
    
    return torch.tensor(np.stack(vector_field_values, axis=1), dtype=torch.float32, device=device)

# Training function
def train_potential_reconstruction(sympy_vector_field, num_points=1000, epochs=2000, device="cpu"):
    # Generate random points in the domain
    points = np.random.uniform(-1, 1, (num_points, 3))
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    # Convert sympy vector field to torch tensor
    vector_field_tensor = convert_sympy_to_torch(sympy_vector_field, points, device=device)
    
    # Create and train model
    model = PotentialNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # List to store loss values for plotting
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute gradient of the potential function
        predicted_vector_field = model.gradient(points_tensor)
        
        # Loss is the difference between predicted and actual vector field
        loss = torch.mean((predicted_vector_field - vector_field_tensor) ** 2)
        
        # Store loss value
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()
    
    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Log scale often better visualizes loss convergence
    plt.grid(True)
    
    # # Save the figure with timestamp
    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"loss_plot_{timestamp}.jpg"
    # save_path = "../figures/" + filename
    # plt.savefig(save_path)
    # print(f"Loss plot saved to {save_path}")
    
    plt.show()
    
    return model, loss_history

# Example usage
if __name__ == "__main__":
    # Define a sympy vector field (example: gradient of f(x,y,z) = x^2 + y^2 + z^2)
    x, y, z = sp.symbols('x y z')
    # Original scalar function (this is what we want to reconstruct)
    scalar_function = x**2 + y**2 + z**2
    
    # Vector field (gradient of scalar function)
    vector_field = [sp.diff(scalar_function, x), 
                    sp.diff(scalar_function, y),
                    sp.diff(scalar_function, z)]
    
    # Add some noise to simulate "almost curl-free"
    noise_level = 0.05
    noisy_vector_field = [component + noise_level * sp.sin(x*y*z) for component in vector_field]
    
    # Train the model to reconstruct the scalar function
    model, loss_history = train_potential_reconstruction(
        noisy_vector_field, 
        num_points=2000, 
        epochs=8000, 
        device=device)
    
    # The trained model now approximates the scalar potential function
    
    # Additional visualization: compare true vs. predicted on a grid
    print("Generating comparison visualization...")
    
    # Create a grid of points for visualization (on z=0 plane for simplicity)
    grid_size = 20
    x_range = np.linspace(-1, 1, grid_size)
    y_range = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)  # Z = 0 plane
    
    grid_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    
    # Get predictions from the model
    with torch.no_grad():
        predicted_potential = model(grid_tensor).cpu().numpy().reshape(grid_size, grid_size)
    
    # Calculate the true potential values (up to a constant)
    true_potential = X**2 + Y**2  # Z=0 so this is the true function on this plane
    
    # Normalize both for better comparison
    predicted_potential = predicted_potential - predicted_potential.min()
    predicted_potential = predicted_potential / predicted_potential.max()
    true_potential = true_potential - true_potential.min()
    true_potential = true_potential / true_potential.max()
    
    # Plot the 2D comparison
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, true_potential, levels=20)
    plt.colorbar(label='Potential Value')
    plt.title('True Potential Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, predicted_potential, levels=20)
    plt.colorbar(label='Potential Value')
    plt.title('Predicted Potential Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    # plt.savefig('../figures/potential_comparison.jpg')
    plt.show()
    
    # Create a 1D comparison along x-axis (with y=0, z=0)
    x_1d = np.linspace(-1, 1, 100)
    y_1d = np.zeros_like(x_1d)
    z_1d = np.zeros_like(x_1d)
    points_1d = np.vstack([x_1d, y_1d, z_1d]).T
    points_1d_tensor = torch.tensor(points_1d, dtype=torch.float32, device=device)
    
    # Get predictions for 1D points
    with torch.no_grad():
        predicted_values_1d = model(points_1d_tensor).cpu().numpy().flatten()
    
    # Calculate true function values (adjust constant to align with predicted)
    true_values_1d = x_1d**2  # When y=0, z=0, the function is x^2
    
    # Adjust the predicted values to match the scale of the true values
    # Find scaling factor by least squares
    scale = np.sum(true_values_1d * predicted_values_1d) / np.sum(predicted_values_1d**2)
    adjusted_predicted = scale * predicted_values_1d
    
    # Calculate error metrics
    mse = np.mean((true_values_1d - adjusted_predicted)**2)
    mae = np.mean(np.abs(true_values_1d - adjusted_predicted))
    
    # Plot 1D comparison
    plt.figure(figsize=(10, 6))
    plt.plot(x_1d, true_values_1d, 'b-', linewidth=2, label='True Function')
    plt.plot(x_1d, adjusted_predicted, 'r--', linewidth=2, label='Predicted Function')
    plt.title(f'True vs Predicted Function Values (y=0, z=0)\nMSE: {mse:.6f}, MAE: {mae:.6f}')
    plt.xlabel('X')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Create a 3D scatter plot to compare true vs predicted values
    num_test_points = 500
    test_points = np.random.uniform(-1, 1, (num_test_points, 3))
    test_tensor = torch.tensor(test_points, dtype=torch.float32, device=device)
    
    # Get predictions for random 3D points
    with torch.no_grad():
        predicted_values_3d = model(test_tensor).cpu().numpy().flatten()
    
    # Calculate true function values
    true_values_3d = test_points[:, 0]**2 + test_points[:, 1]**2 + test_points[:, 2]**2
    
    # Find scaling factor for 3D comparison
    scale_3d = np.sum(true_values_3d * predicted_values_3d) / np.sum(predicted_values_3d**2)
    adjusted_predicted_3d = scale_3d * predicted_values_3d
    
    # Calculate error metrics for 3D comparison
    mse_3d = np.mean((true_values_3d - adjusted_predicted_3d)**2)
    mae_3d = np.mean(np.abs(true_values_3d - adjusted_predicted_3d))
    
    # Create scatter plot comparing true vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values_3d, adjusted_predicted_3d, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(np.max(true_values_3d), np.max(adjusted_predicted_3d))
    min_val = min(np.min(true_values_3d), np.min(adjusted_predicted_3d))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.title(f'True vs Predicted Function Values (3D)\nMSE: {mse_3d:.6f}, MAE: {mae_3d:.6f}')
    plt.xlabel('True Function Value')
    plt.ylabel('Predicted Function Value')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
