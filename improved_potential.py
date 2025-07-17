import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
x0 = np.array([1.0, -0.2, 0.2])

x1, x2, x3 = sp.symbols('x1 x2 x3')
variable_x = sp.Matrix([x1, x2, x3])
n_dim = variable_x.shape[0]

# Original system dynamics
fx = sp.Matrix([
    -x1,
    -2*x2 - x1*x3,
    3*x1*x2
])

gx = sp.Matrix([
    1,
    0,
    0
])

h = 1.98789561912197*x2**2 - 0.66263184896675*x3**2

# Compute Lie derivatives for initial z-coordinates
Lfh = (h.diff(variable_x).T*fx)[0]
Lf2h = (Lfh.diff(variable_x).T*fx)[0]
Lf3h = (Lf2h.diff(variable_x).T*fx)[0]
LgLf2h = (Lf2h.diff(variable_x).T*gx)[0]

# Create lambda functions for numerical evaluation
h_func = sp.lambdify((x1, x2, x3), h, "numpy")
Lfh_func = sp.lambdify((x1, x2, x3), Lfh, "numpy")
Lf2h_func = sp.lambdify((x1, x2, x3), Lf2h, "numpy")
Lf3h_func = sp.lambdify((x1, x2, x3), Lf3h, "numpy")
LgLf2h_func = sp.lambdify((x1, x2, x3), LgLf2h, "numpy")

# TODO if sinusoidal don't work, change it to constant
def sinusoidal_v_input(t):
    """Sinusoidal control input v for the linearized system"""
    return 1 * np.sin(1.0 * t)
    # return 1.0

def compute_initial_z_coordinates(h_expr, fx, variable_x, x0_val, n_dim):
    """
    Compute initial z-coordinates for n-dimensional linearized system
    
    Args:
        h_expr: symbolic output function h(x)
        fx: symbolic vector field f(x)
        variable_x: symbolic variable vector
        x0_val: initial state values
        n_dim: desired dimension of z-coordinates
    
    Returns:
        z0: initial z-coordinates array
    """
    # Compute Lie derivatives up to order n_dim-1
    lie_derivatives = [h_expr]  # L^0_f h = h
    
    for i in range(n_dim - 1):
        # Compute L^{i+1}_f h
        lie_deriv = (lie_derivatives[i].diff(variable_x).T * fx)[0]
        lie_derivatives.append(lie_deriv)
    
    # Create lambda functions and evaluate at x0
    z0 = np.zeros(n_dim)
    for i in range(n_dim):
        lie_func = sp.lambdify(tuple(variable_x), lie_derivatives[i], "numpy")
        z0[i] = lie_func(*x0_val)
    
    return z0

# For 3D (current case):
z0_3d = compute_initial_z_coordinates(h, fx, variable_x, x0, n_dim)  # Already computed above


# Use the 3D case for this simulation
z0_current = z0_3d
print(f"Initial z-coordinates: z0 = {z0_current}")
print(f"Dimension of z: {len(z0_current)}")

def original_system(t, x):

    n = len(x) # Number of variables

    Lf3h_val = Lf3h_func(x[0], x[1], x[2])
    LgLf2h_val = LgLf2h_func(x[0], x[1], x[2])

    u_val = (sinusoidal_v_input(t) - Lf3h_val)/LgLf2h_val

    # Original system dynamics
    dx1dt = -x[0] + u_val
    dx2dt = -2*x[1] - x[0]*x[2]
    dx3dt = 3*x[0]*x[1]
    
    return np.array([dx1dt, dx2dt, dx3dt])



def linearised_system(t, z):
    """
    Linearised system dynamics in z-coordinates with input v
    
    Args:
        t: time
        z: state vector of dimension n
    
    Returns:
        dz: derivative of state vector
    """
    n = len(z)
    
    # Get the control input
    v = sinusoidal_v_input(t)
    
    # Initialize derivative vector
    dz = np.zeros(n)
    
    # Chain of integrators: ż_i = z_{i+1} for i = 1, ..., n-1
    for i in range(n-1):
        dz[i] = z[i+1]
    
    # Last equation: żₙ = v(t)
    dz[n-1] = v
    
    return dz

print("Simulating original system...")
sol_ori = solve_ivp(
    original_system,
    t_span,
    x0,
    t_eval = t_eval,
    method='RK45'
)

z_ori = np.zeros((n_dim, len(sol_ori.t)))
for i in range(len(sol_ori.t)):
    x_val = sol_ori.y[:, i]
    z_ori[0, i] = h_func(*x_val)
    z_ori[1, i] = Lfh_func(*x_val)
    z_ori[2, i] = Lf2h_func(*x_val)

print("Simulating linearised system...")
sol_lin = solve_ivp(
    linearised_system,
    t_span,
    z0_current,
    t_eval=t_eval,
    method='RK45'
)

plt.figure(figsize=(10, 8))
color_map = ['r', 'g', 'b', 'm', 'o']
plt.subplot(2, 2, 1)
for i in range(n_dim):
    plt.plot(sol_lin.t, sol_lin.y[i], color_map[i], label=f"z{i+1} (linearised)")
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Linearised system states')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
for i in range(n_dim):
    plt.plot(sol_ori.t, sol_ori.y[i], color_map[i], label=f"x{i+1} (original)")
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Original system states')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
for i in range(n_dim):
    plt.plot(sol_ori.t, z_ori[i], color_map[i], label=f"z{i+1} (original)")
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Original systems states (transformed)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Simulation completed successfully!")
print(f"Final z-coordinates: {sol_lin.y[:, -1]}")
print(f"System dimension: {len(z0_current)}")