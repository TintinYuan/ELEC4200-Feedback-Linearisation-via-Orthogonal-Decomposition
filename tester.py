import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils import zero_small_coefficients

# Current date and time information
print(f"Simulation executed on: 2025-07-16 07:36:05 UTC")
print(f"User: TintinYuan")

# Define symbolic variables
x1, x2, x3 = sp.symbols('x1 x2 x3')
u = sp.symbols('u')

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

# Output function
a = 1; b = 2; c = 1; theta = 3; k = 2; omega0 = 1
h = -1.98789561912197*x2**2 - 0.66263184896675*x3**2
h = 1.0*(theta * x2**2 + c * x3**2) + (-theta*(k/b)**2 - c * omega0)# NOTE test output function

# Compute Lie derivatives
Lfh = h.diff(x1)*fx[0] + h.diff(x2)*fx[1] + h.diff(x3)*fx[2]
Lf2h = Lfh.diff(x1)*fx[0] + Lfh.diff(x2)*fx[1] + Lfh.diff(x3)*fx[2]
Lf3h = Lf2h.diff(x1)*fx[0] + Lf2h.diff(x2)*fx[1] + Lf2h.diff(x3)*fx[2]

Lgh = h.diff(x1)*gx[0] + h.diff(x2)*gx[1] + h.diff(x3)*gx[2]
LgLfh = Lfh.diff(x1)*gx[0] + Lfh.diff(x2)*gx[1] + Lfh.diff(x3)*gx[2]
LgLf2h = Lf2h.diff(x1)*gx[0] + Lf2h.diff(x2)*gx[1] + Lf2h.diff(x3)*gx[2]

# TAG clean trival zero coefficient
Lgh = zero_small_coefficients(sp.expand(Lgh))
LgLfh = zero_small_coefficients(sp.expand(LgLfh))
LgLf2h = zero_small_coefficients(sp.expand(LgLf2h))

# Create lambda functions for numerical evaluation
h_func = sp.lambdify((x1, x2, x3), h, "numpy")
Lfh_func = sp.lambdify((x1, x2, x3), Lfh, "numpy")
Lf2h_func = sp.lambdify((x1, x2, x3), Lf2h, "numpy")
Lf3h_func = sp.lambdify((x1, x2, x3), Lf3h, "numpy")
LgLf2h_func = sp.lambdify((x1, x2, x3), LgLf2h, "numpy")

print("\nSystem Analysis:")
print(f"Output function h(x) = {h}")
print(f"Lfh(x) = {sp.simplify(Lfh)}")
print(f"Lf²h(x) = {sp.simplify(Lf2h)}")
print(f"Lf³h(x) = {sp.simplify(Lf3h)}")
print(f"LgLf²h(x) = {sp.simplify(LgLf2h)}")

# Check relative degree
if Lgh == 0 and LgLfh == 0 and LgLf2h != 0:
    print("\nSystem has relative degree 3 (full state feedback linearization is possible)")
else:
    print("\nWarning: System doesn't have relative degree 3")

# TAG Define sinusoidal input for the original system
def sinusoidal_input(t):
    # return 20.0 * np.exp(-t) + 0.1
    return 1.0 * np.sin(1.0 * t)

# Function to compute v = Lf³h(x) + LgLf²h(x)*u
def compute_v(x, u_val):
    x1_val, x2_val, x3_val = x
    
    # Compute Lf³h and LgLf²h at the current state
    Lf3h_val = Lf3h_func(x1_val, x2_val, x3_val)
    LgLf2h_val = LgLf2h_func(x1_val, x2_val, x3_val)
    
    # v = Lf³h(x) + LgLf²h(x)*u
    v_val = Lf3h_val + LgLf2h_val * u_val
    
    return v_val

# Function to compute u = (v - Lf³h(x))/LgLf²h(x)
def compute_u(x, v_val):
    x1_val, x2_val, x3_val = x
    
    # Compute Lf³h and LgLf²h at the current state
    Lf3h_val = Lf3h_func(x1_val, x2_val, x3_val)
    LgLf2h_val = LgLf2h_func(x1_val, x2_val, x3_val)
    
    # Avoid division by zero
    if abs(LgLf2h_val) < 1e-10:
        return 0
    
    # u = (v - Lf³h(x))/LgLf²h(x)
    u_val = (v_val - Lf3h_val) / LgLf2h_val
    
    return u_val

# Original system dynamics with sinusoidal input
def original_system(t, x):
    x1_val, x2_val, x3_val = x
    
    # Apply sinusoidal control input
    u_val = sinusoidal_input(t)

    # Transformn input from v to u
    # v_val = sinusoidal_input(t)
    # u_val = compute_u(x, v_val)
    
    # Original system dynamics
    dx1dt = -x1_val + u_val
    dx2dt = -2*x2_val - x1_val*x3_val
    dx3dt = 3*x1_val*x2_val
    
    return np.array([dx1dt, dx2dt, dx3dt])

# Linearized system dynamics with transformed input
def linearized_system(t, z, original_sol):
    # At time t, get the corresponding x from the original system
    # We need this to compute the correct v
    
    # Find the closest time point in original_sol.t
    idx = np.argmin(np.abs(original_sol.t - t))
    x_val = np.array([original_sol.y[0, idx], original_sol.y[1, idx], original_sol.y[2, idx]])
    
    # Transform the input
    u_val = sinusoidal_input(t)
    v_val = compute_v(x_val, u_val)
    
    # Chain of integrators dynamics
    z1, z2, z3 = z
    dz1dt = z2
    dz2dt = z3
    dz3dt = v_val
    
    return np.array([dz1dt, dz2dt, dz3dt])

# Simulation parameters
t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 10000)
x0 = np.array([1.0, 0.5, 0.2])

# First, simulate the original system
sol_orig = solve_ivp(
    original_system,
    t_span,
    x0,
    t_eval=t_eval,
    method='RK45'
)

# Initial z-coordinates (transformed from initial x-coordinates)
z0 = np.array([
    h_func(*x0),
    Lfh_func(*x0),
    Lf2h_func(*x0)
])

# Then, simulate the linearized system using transformed inputs from original system
sol_lin = solve_ivp(
    lambda t, z: linearized_system(t, z, sol_orig),
    t_span,
    z0,
    t_eval=t_eval,
    method='RK45'
)

# Calculate the z-coordinates from the original system trajectory
z_trajectory = np.zeros((3, len(sol_orig.t)))

for i in range(len(sol_orig.t)):
    x_val = [sol_orig.y[0, i], sol_orig.y[1, i], sol_orig.y[2, i]]
    z_trajectory[0, i] = h_func(*x_val)
    z_trajectory[1, i] = Lfh_func(*x_val)
    z_trajectory[2, i] = Lf2h_func(*x_val)

# Calculate the control inputs
u_vals = np.array([sinusoidal_input(t) for t in sol_orig.t])
v_vals = np.zeros_like(u_vals)

for i in range(len(sol_orig.t)):
    x_val = [sol_orig.y[0, i], sol_orig.y[1, i], sol_orig.y[2, i]]
    v_vals[i] = compute_v(x_val, u_vals[i])

# v_vals = np.array([sinusoidal_input(t) for t in sol_lin.t])
# u_vals = np.zeros_like(v_vals)
# for i in range(len(sol_orig.t)):
#     x_val =  [sol_orig.y[0, i], sol_orig.y[1, i], sol_orig.y[2, i]]
#     u_vals[i] = compute_u(x_val, v_vals[i])

# SUPTAG Plot the results
plt.figure(figsize=(10, 8))

# Plot original system states
plt.subplot(2, 2, 1)
plt.plot(sol_orig.t, sol_orig.y[0], 'r', label='x₁')
plt.plot(sol_orig.t, sol_orig.y[1], 'g', label='x₂')
plt.plot(sol_orig.t, sol_orig.y[2], 'b', label='x₃')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Original System States with Sinusoidal Input')
plt.grid(True)
plt.legend()

# Plot transformed states from original system
plt.subplot(2, 2, 2)
plt.plot(sol_orig.t, z_trajectory[0], 'r', label='z₁ = h(x)')
plt.plot(sol_orig.t, z_trajectory[1], 'g', label='z₂ = Lfh(x)')
plt.plot(sol_orig.t, z_trajectory[2], 'b', label='z₃ = Lf²h(x)')
plt.xlabel('Time t')
plt.ylabel('Transformed States')
plt.title('Transformed States from Original System')
plt.grid(True)
plt.legend()

# Plot linearized system states
plt.subplot(2, 2, 3)
plt.plot(sol_lin.t, sol_lin.y[0], 'r', label='z₁')
plt.plot(sol_lin.t, sol_lin.y[1], 'g', label='z₂')
plt.plot(sol_lin.t, sol_lin.y[2], 'b', label='z₃')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Linearized System States with Transformed Input')
plt.grid(True)
plt.legend()

# Plot control inputs
plt.subplot(2, 2, 4)
plt.plot(sol_orig.t, u_vals, 'r', label='Original input u(t)')
plt.plot(sol_orig.t, v_vals, 'b', label='Transformed input v(t)')
plt.xlabel('Time t')
plt.ylabel('Control Inputs')
plt.title('Control Inputs Comparison')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('exact_trajectory_comparison_part1.png')
plt.show()

# Create a direct comparison of each state
plt.figure(figsize=(9, 3))

# Compare z₁
plt.subplot(1, 3, 1)
plt.plot(sol_orig.t, z_trajectory[0], 'r', label='z₁ from original')
plt.plot(sol_lin.t, sol_lin.y[0], 'b--', label='z₁ from linearized')
plt.xlabel('Time t')
plt.ylabel('z₁')
plt.title('Comparison of z₁')
plt.grid(True)
plt.legend()

# Compare z₂
plt.subplot(1, 3, 2)
plt.plot(sol_orig.t, z_trajectory[1], 'r', label='z₂ from original')
plt.plot(sol_lin.t, sol_lin.y[1], 'b--', label='z₂ from linearized')
plt.xlabel('Time t')
plt.ylabel('z₂')
plt.title('Comparison of z₂')
plt.grid(True)
plt.legend()

# Compare z₃
plt.subplot(1, 3, 3)
plt.plot(sol_orig.t, z_trajectory[2], 'r', label='z₃ from original')
plt.plot(sol_lin.t, sol_lin.y[2], 'b--', label='z₃ from linearized')
plt.xlabel('Time t')
plt.ylabel('z₃')
plt.title('Comparison of z₃')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('exact_trajectory_comparison_part2.png')
plt.show()

# Calculate errors between trajectories
errors = np.zeros((3, len(sol_lin.t)))
for i in range(3):
    # Interpolate z_trajectory to match sol_lin.t time points
    z_interp = np.interp(sol_lin.t, sol_orig.t, z_trajectory[i])
    errors[i] = z_interp - sol_lin.y[i]

# Plot errors
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(sol_lin.t, errors[0])
plt.xlabel('Time t')
plt.ylabel('Error in z₁')
plt.title('Error: z₁ from original - z₁ from linearized')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(sol_lin.t, errors[1])
plt.xlabel('Time t')
plt.ylabel('Error in z₂')
plt.title('Error: z₂ from original - z₂ from linearized')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(sol_lin.t, errors[2])
plt.xlabel('Time t')
plt.ylabel('Error in z₃')
plt.title('Error: z₃ from original - z₃ from linearized')
plt.grid(True)

plt.tight_layout()
plt.savefig('trajectory_errors.png')
plt.show()

# Calculate statistics
print("\nError Analysis:")
for i in range(3):
    mean_err = np.mean(np.abs(errors[i]))
    max_err = np.max(np.abs(errors[i]))
    print(f"z{i+1} - Mean absolute error: {mean_err:.6f}, Max absolute error: {max_err:.6f}")

# Calculate correlation coefficients between trajectories
print("\nCorrelation Analysis:")
for i in range(3):
    # Interpolate z_trajectory to match sol_lin.t time points
    z_interp = np.interp(sol_lin.t, sol_orig.t, z_trajectory[i])
    corr = np.corrcoef(z_interp, sol_lin.y[i])[0, 1]
    print(f"Correlation coefficient for z{i+1}: {corr:.8f}")

print("\nNote: Small numerical errors may occur due to interpolation and numerical integration,")
print("but theoretically the trajectories should be identical in the ideal case.")