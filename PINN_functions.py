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

 