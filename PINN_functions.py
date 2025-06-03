import torch

# --- Scalar potential functions and their gradient vector fields

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


def compute_gradient(vector_field, points, delta = 1e-5):
    """
    Compute gradient using finite differences
    vector_field: (N, 3) tensor
    points: (N, 3) tensor
    """

    N = points.shape[0] # how many samples in total 
    m = points.shape[1] # dimension of each point
    if N < 2:
        return torch.zeros((N, m), device=points.device)
    
    # Simple central difference approximation
    div = torch.zeros((N, m), device=points.device)

    for i in range(N):
        if i > 1 and i < N-1:
            # Use neighbouring points for finite difference
            dx = points[i+1, 0] - points[i-1, 0] + delta
            dy = points[i+1, 1] - points[i-1, 1] + delta
            dz = points[i+1, 2] - points[i-1, 2] + delta

            dvx_dx = (vector_field[i+1, 0] - vector_field[i-1, 0])/dx
            dvy_dy = (vector_field[i+1, 1] - vector_field[i-1, 1])/dy
            dvz_dz = (vector_field[i+1, 2] - vector_field[i-1, 2])/dz

            div[i, :] = torch.tensor([dvx_dx, dvy_dy, dvz_dz], device=points.device)


    return div


def compute_gradient_of_scalar(scalar_field, points, h=1e-4):
    """
    Compute gradient of a scalar field using finite differences
    scalar_field: (N,) tensor - scalar values at each point
    points: (N, 3) tensor - spatial coordinates
    Returns: (N, 3) tensor - gradient vectors
    """
    N = points.shape[0]
    grad_scalar = torch.zeros(N, 3, device=points.device)
    
    if N < 3:
        return grad_scalar
    
    for i in range(1, N-1):  # Use central differences where possible
        # Find nearest neighbors (this is a simplified approach)
        # In practice, you might want to use proper spatial neighbors
        
        # Approximate gradients using adjacent points
        for dim in range(3):  # x, y, z dimensions
            if i > 0 and i < N-1:
                dx = points[i+1, dim] - points[i-1, dim] + eps
                df = scalar_field[i+1] - scalar_field[i-1]
                grad_scalar[i, dim] = df / dx
    
    return grad_scalar

def v_pytorch_batch(v_pytorch_single, x_batch):
    """Batch version of vector field"""
    return torch.func.vmap(v_pytorch_single)(x_batch)


# %% Function below for testing behaviour

class CurlFreePKVLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, K_output, x_batch, v_batch, grad_v_batch):
        """
        Calculates the loss for p*v to be curl-free, where p = exp(K_output).

        Args: 
            K_output (Tensor): Output of the network K(x) for the batch, shape (N, 1) or (N, ). N is the batch size
            x_batch (Tensor): Input coordinates for the batch, shape (N, n_dims). Must have requires_grad=True.
            v_batch (Tensor): Known vector field v(x) evaluated at x_batch, shape (N, n_dims).
            grad_v_batch (Tensor): Jacobian of v(x) evaluated at x_batch, where grad_v_batch[s, i, j] is dv_i/dx_j for sample s. So it's three dimensional, shape (N, n_dims, d_dims)

        Returns:
            (Tensor): Scalar loss value.
        """

        N, n_dims = x_batch.shape

        if K_output.ndim == 1:
            K_output = K_output.unsqueeze(1) # Ensure K_output is (N, 1)

        # Compute gradients of K w.r.t. x_batch (dK/dx_i)
        grad_K_batch = torch.autograd.grad(
            outputs=K_output,
            inputs=x_batch,
            grad_outputs=torch.ones_like(K_output),
            create_graph=True,
            retain_graph=True
        )[0]

        term1_dKdxj_vi = torch.einsum('sj,si->sij', grad_K_batch, v_batch)

        T_tensor = term1_dKdxj_vi + grad_v_batch

        R_tensor = T_tensor - T_tensor.transpose(1, 2)

        squared_residuals = R_tensor**2

        if n_dims < 2: # No off-diagonal element to sum
            return torch.tensor(0.0, device=x_batch.device, required_grad=True)
        
        # generates the indices of the upper triangular part of a 2D matrix
        row_indices, col_indices = torch.triu_indices(n_dims, n_dims, offset=1)

        selected_squared_residuals = squared_residuals[:, row_indices, col_indices]

        # Select the upper triangle elements for each sample
        sum_sq_residuals_per_sample = torch.sum(selected_squared_residuals, dim=1)

        # Mean loss over the batch
        loss = torch.mean(sum_sq_residuals_per_sample)

        return loss
    

def v_jacobian(v_true, x_points):
    """
    Args: 
        v_true: tensor function of vector field v
        x_points: cloned and isolated x_points for AD operation

    Returns:
        grad_v: Calculated jacobian matrix of v
    """
    
    jacobian = torch.func.jacrev(v_true)
    grad_v = torch.func.vmap(jacobian)(x_points)

    return grad_v