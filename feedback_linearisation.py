import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime

# Add timestamp for result saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

t_span = (0, 0.4)
t_eval = np.linspace(t_span[0], t_span[1], 100)
x0 = np.array([1, 1, 2])

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

def sinusoidal_v_input(t):
    """Sinusoidal control input v for the linearized system"""
    return 0.0 * np.sin(1.0 * t)
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

# Function to convert x coordinates to z coordinates
def x_to_z(x_vals):
    """Convert x coordinates to z coordinates"""
    if x_vals.ndim == 1:
        return np.array([
            h_func(x_vals[0], x_vals[1], x_vals[2]),
            Lfh_func(x_vals[0], x_vals[1], x_vals[2]),
            Lf2h_func(x_vals[0], x_vals[1], x_vals[2])
        ])
    else:
        z_vals = np.zeros((x_vals.shape[0], 3))
        for i in range(x_vals.shape[0]):
            z_vals[i, 0] = h_func(x_vals[i, 0], x_vals[i, 1], x_vals[i, 2])
            z_vals[i, 1] = Lfh_func(x_vals[i, 0], x_vals[i, 1], x_vals[i, 2])
            z_vals[i, 2] = Lf2h_func(x_vals[i, 0], x_vals[i, 1], x_vals[i, 2])
        return z_vals

# Simulate both systems
print("Simulating original system...")
sol_original = solve_ivp(original_system, t_span, x0, t_eval=t_eval, method='RK45')

print("Simulating linearized system...")
sol_linearized = solve_ivp(linearised_system, t_span, z0_current, t_eval=t_eval, method='RK45')

# Convert original system trajectory to z-coordinates
print("Converting original system trajectory to z-coordinates...")
z_from_original = x_to_z(sol_original.y.T)

# Create comparison plots
plt.figure(figsize=(15, 10))

# Plot z1 comparison
plt.subplot(2, 2, 1)
plt.plot(t_eval, z_from_original[:, 0], 'b-', label='Original system (z₁)', linewidth=2)
plt.plot(t_eval, sol_linearized.y[0, :], 'r--', label='Linearized system (z₁)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('z₁')
plt.title('z₁ Coordinate Comparison')
plt.legend()
plt.grid(True)

# Plot z2 comparison
plt.subplot(2, 2, 2)
plt.plot(t_eval, z_from_original[:, 1], 'b-', label='Original system (z₂)', linewidth=2)
plt.plot(t_eval, sol_linearized.y[1, :], 'r--', label='Linearized system (z₂)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('z₂')
plt.title('z₂ Coordinate Comparison')
plt.legend()
plt.grid(True)

# Plot z3 comparison
plt.subplot(2, 2, 3)
plt.plot(t_eval, z_from_original[:, 2], 'b-', label='Original system (z₃)', linewidth=2)
plt.plot(t_eval, sol_linearized.y[2, :], 'r--', label='Linearized system (z₃)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('z₃')
plt.title('z₃ Coordinate Comparison')
plt.legend()
plt.grid(True)

# Plot control input
plt.subplot(2, 2, 4)
v_vals = [sinusoidal_v_input(t) for t in t_eval]
plt.plot(t_eval, v_vals, 'g-', label='Input v(t)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('v(t)')
plt.title('Control Input v(t)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'feedback_linearization_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and display errors
max_errors = np.zeros(3)
rms_errors = np.zeros(3)

for i in range(3):
    error = np.abs(z_from_original[:, i] - sol_linearized.y[i, :])
    max_errors[i] = np.max(error)
    rms_errors[i] = np.sqrt(np.mean(error**2))

print("\n" + "="*60)
print("FEEDBACK LINEARIZATION VERIFICATION RESULTS")
print("="*60)
print(f"Time span: {t_span[0]} to {t_span[1]} seconds")
print(f"Initial conditions (x): {x0}")
print(f"Initial conditions (z): {z0_current}")
print("\nOutput function h(x):")
print(f"h = {h}")
print(f"\nCoordinate transformations:")
print(f"z₁ = h(x) = {h}")
print(f"z₂ = Lₓh(x)")
print(f"z₃ = L²ₓh(x)")
print("\n" + "-"*40)
print("ERROR ANALYSIS:")
print("-"*40)
for i in range(3):
    print(f"z₁ coordinate:")
    print(f"  Maximum error: {max_errors[i]:.6e}")
    print(f"  RMS error: {rms_errors[i]:.6e}")

# Check if errors are within acceptable tolerance
tolerance = 1e-6
all_within_tolerance = all(max_errors < tolerance)

print(f"\nTolerance check (< {tolerance}):")
if all_within_tolerance:
    print("✅ VERIFICATION SUCCESSFUL: Both systems behave identically!")
    print("   The feedback linearization is working correctly.")
else:
    print("❌ VERIFICATION FAILED: Systems show significant differences.")
    print("   Check the feedback linearization implementation.")

print("\n" + "="*60)

