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

def v_true_tensor(x_single_tensor):
    """The true conservative vector field (gradient of h_true)."""
    # Compute partial derivatives symbolically or manually
    x = x_single_tensor[0]
    y = x_single_tensor[1]
    z = x_single_tensor[2]

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

    dv_dx = 0 * x
    dv_dy = c1 * 2 * theta * y
    dv_dz = c1 * 2 * c * z
    return torch.stack([dv_dx, dv_dy, dv_dz], dim=-1)

def v_true2_tensor(x_single_tensor):

    x = x_single_tensor[0]
    y = x_single_tensor[1]
    z = x_single_tensor[2]

    c1 = 2.0
    c2 = 1.0
    c = 3.0
    theta = 1.44
    k = 1.0

    dv_dx = 0 * x
    dv_dy = c1 * 2 * theta * y
    dv_dz = c1 * 2 * c * z
    return torch.stack([dv_dx, dv_dy, dv_dz], dim=-1)

def v_pytorch_batch(v_pytorch_single, x_batch):
    """Batch version of vector field"""
    return torch.func.vmap(v_pytorch_single)(x_batch)

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


def compute_gradient_of_scalar(scalar_field, points, epsilon=1e-4):
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
                dx = points[i+1, dim] - points[i-1, dim] + epsilon
                df = scalar_field[i+1] - scalar_field[i-1]
                grad_scalar[i, dim] = df / dx
    
    return grad_scalar

# Enhanced accuracy metrics
def compute_proportionality_metrics(grad_h, v_true, eps=1e-6):
    # Cross product magnitude (should be 0 for proportional vectors)
    cross_prod = torch.cross(grad_h, v_true, dim=1)
    cross_magnitude = torch.norm(cross_prod, dim=1)
    mean_cross_error = torch.mean(cross_magnitude).item()

    # Cosine similarity (should be \pm 1)
    grad_h_norm = torch.norm(grad_h, dim=1)
    v_norm = torch.norm(v_true, dim=1)

    valid_mask = (grad_h_norm > eps) & (v_norm > eps)
    if valid_mask.sum() > 0:
        cosine_sim = torch.sum(grad_h[valid_mask] * v_true[valid_mask], dim=1)/(grad_h_norm[valid_mask] * v_norm[valid_mask])
        mean_cosine_error = torch.mean((torch.abs(cosine_sim) - 1)**2).item()
    else:
        mean_cosine_error = float('inf')

    return mean_cross_error, mean_cosine_error


def generate_training_data(domain, num_train_points):
    """Generate random training samples"""
    [x_min, x_max], [y_min, y_max], [z_min, z_max] = domain


    coords = torch.rand(num_train_points, 3)
    coords[:, 0] = coords[:, 0] * (x_max - x_min) + x_min
    coords[:, 1] = coords[:, 1] * (y_max - y_min) + y_min
    coords[:, 2] = coords[:, 2] * (z_max - z_min) + z_min

    return coords


# SUPTAG Loss functions go below:

class v2grad_CurlFreePKVLoss(torch.nn.Module):
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

        if K_output.ndim == 1: # number of dimension
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
    
# TAG Curl-free v2grad loss
class v2grad_CurlFreepvLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, K_output, x_batch, v_batch, grad_v_batch):
        """
        Calculates the p*v to be curl-free, with a fixed point value K(x_0) = 1

        Args:
            K_output (Tensor): Output of the network K(x) for the batch, shape (N, 1) or (N, ). N is the batch size
            x_batch (Tensor): Input coordinates for the batch, shape (N, n_dims). Must have requires_grad=True.
            v_batch (Tensor): Known vector field v(x) evaluated at x_batch, shape (N, n_dims).
            grad_v_batch (Tensor): Jacobian of v(x) evaluated at x_batch, where grad_v_batch[s, i, j] is dv_i/dx_j for sample s. So it's three dimensional, shape (N, n_dims, d_dims)

        Returns:
            (Tensor): Scalar loss value.
        """

        N, n_dims = x_batch.shape

        if K_output.ndim == 1: # number of dimension
            K_output = K_output.unsqueeze(1)

        grad_K_batch = torch.autograd.grad(
            outputs=K_output,
            inputs=x_batch,
            grad_outputs=torch.ones_like(K_output), # if the output is a scalar, it doesn't change anything
            create_graph=True,
            retain_graph=True # if computing only first-order gradients, this can be set false
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
    
# TAG point constraint v2grad loss
class v2grad_PointConstraintLoss(torch.nn.Module):
    def __init__(self, constraint_points, constraint_values):
        """
        Args: 
            constraint_points (Tensor or List): Points where constraints apply, shape (N, n_dims)
            constraint_values (Tensor or List): Target values at constraint points, shape (N, )
        """
        super().__init__()
        if isinstance(constraint_points, list):
            constraint_points = torch.tensor(constraint_points, dtype=torch.float32)
        if isinstance(constraint_values, list):
            constraint_values = torch.tensor(constraint_values, dtype=torch.float32)

        self.register_buffer('constraint_points', constraint_points)
        self.register_buffer('constraint_values', constraint_values)

    def forward(self, model):
        """
        Args:
            model: Neural network model

        Returns:
            Constraint loss
        """
        # Get device from model's first parameter
        device = next(model.parameters()).device

        # Move constraint_points to same device
        constraint_points = self.constraint_points.to(device).clone().detach().requires_grad_(True)
        constraint_values = self.constraint_values.to(device)
        model_output = model(constraint_points)

        if model_output.ndim == 2 and model_output.shape[1] == 1:
            model_output = model_output.squeeze(1)

        constraint_loss = torch.mean((model_output - constraint_values)**2)
        return constraint_loss
    
# TAG v2func cross loss
class v2func_cross_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, K_output, x_batch, v_batch):
        """
        Calculate the proportional loss of the neural network gradient h_grad with the known v

        Args:
            K_output (Tensor): Output of the network K(x) for the batch, shape (N, 1) or (N, ). N is the batch size
            x_batch (Tensor): Input coordinates for the batch, shape (N, n_dims). Must have requires_grad=True.
            v_batch (Tensor): Known vector field v(x) evaluated at x_batch, shape (N, n_dims).

        Returns:
            (Tensor): Scalar loss value.
        """

        if K_output.ndim == 1: # number of dimension
            K_output = K_output.unsqueeze(1) # Ensure K_output is (N, 1)

        grad_h = torch.autograd.grad(
            outputs=K_output,
            inputs=x_batch,
            grad_outputs=torch.ones_like(K_output),
            create_graph=True,
            retain_graph=True
        )[0]

        x, y, z = x_batch[:, 0], x_batch[:, 1], x_batch[:, 2]

        cross_product = torch.cross(grad_h, v_batch, dim=1)
        cross_loss = torch.mean(torch.sum(cross_product**2, dim=1))

        return cross_loss
    
class v2func_zero_grad_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, K_output, x_batch):
        """
        Calculate the loss of infinity norm to avoid trivial zero solution

        Args:
            K_output (Tensor): Output of the network K(x) for the batch, shape (N, 1) or (N, ). N is the batch size
            x_batch (Tensor): Input coordinates for the batch, shape (N, n_dims). Must have requires_grad=True.
        Returns:
            (Tensor): Loss value of infinity norm
        """

        if K_output.ndim == 1:
            K_output = K_output.unsqueeze(1)

        grad_h = torch.autograd.grad(
            outputs=K_output,
            inputs=x_batch,
            grad_outputs=torch.ones_like(K_output),
            create_graph=True,
            retain_graph=True
        )[0]

        max_values, _ = torch.max(grad_h, dim=1, keepdim=True)
        max_values = torch.abs(max_values)
        zero_grad_loss = torch.mean(torch.sum(1/max_values, dim=1))

        return zero_grad_loss

