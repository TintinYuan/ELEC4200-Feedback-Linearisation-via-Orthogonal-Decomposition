import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Current date and time information
print(f"Simulation executed on: 2025-07-16 07:21:57 UTC")
print(f"User: TintinYuan")

# Define symbolic variables
x1, x2, x3 = sp.symbols('x1 x2 x3')
variable_x = sp.Matrix([x1, x2, x3])

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
h = 1.98789561912197*x2**2 - 0.66263184896675*x3**2

# Compute Lie derivatives
Lfh = (h.diff(variable_x).T*fx)[0]
Lf2h = (Lfh.diff(variable_x).T*fx)[0]
LgLf2h = (Lf2h.diff(variable_x).T*gx)[0]
Lf3h = (Lf2h.diff(variable_x).T*fx)[0]

# Create lambda functions for numerical evaluation
h_func = sp.lambdify((x1, x2, x3), h, "numpy")
Lfh_func = sp.lambdify((x1, x2, x3), Lfh, "numpy")
Lf2h_func = sp.lambdify((x1, x2, x3), Lf2h, "numpy")
LgLf2h_func = sp.lambdify((x1, x2, x3), LgLf2h, "numpy") 
Lf3h_func = sp.lambdify((x1, x2, x3), Lf3h, "numpy")

print(f"Output function h(x) = {h}")
print(f"Lfh(x) = {sp.simplify(Lfh)}")
print(f"Lf²h(x) = {sp.simplify(Lf2h)}")

# Define the sinusoidal input
def sinusoidal_input(t):
    return 1.0 * np.sin(1.0 * t)

# Original system dynamics with sinusoidal input
def original_system(t, x):
    x1_val, x2_val, x3_val = x
    
    u_val = sinusoidal_input(t)
    
    dx1dt = -x1_val + u_val
    dx2dt = -2*x2_val - x1_val*x3_val
    dx3dt = 3*x1_val*x2_val
    
    return np.array([dx1dt, dx2dt, dx3dt])

# Simulation parameters
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)
x0 = np.array([1.0, 0.5, 0.2])

# Simulate the original system
sol = solve_ivp(
    original_system,
    t_span,
    x0,
    t_eval=t_eval,
    method='RK45'
)

# Calculate the z-coordinates for each point in the trajectory
z_trajectory = np.zeros((3, len(sol.t)))

for i in range(len(sol.t)):
    x_val = [sol.y[0, i], sol.y[1, i], sol.y[2, i]]
    z_trajectory[0, i] = h_func(*x_val)
    z_trajectory[1, i] = Lfh_func(*x_val)
    z_trajectory[2, i] = Lf2h_func(*x_val)

# Create the linearized system dynamics (direct chain of integrators)
# Function to compute v = Lf³h(x) + LgLf²h(x)*u
def compute_v(x, u_val):
    x1_val, x2_val, x3_val = x
    
    # Compute Lf³h and LgLf²h at the current state
    Lf3h_val = Lf3h_func(x1_val, x2_val, x3_val)
    LgLf2h_val = LgLf2h_func(x1_val, x2_val, x3_val)
    
    # v = Lf³h(x) + LgLf²h(x)*u
    v_val = Lf3h_val + LgLf2h_val * u_val
    
    return v_val


def linearized_system(t, z, original_sol):
    z1, z2, z3 = z
    
    # In z-coordinates, input v is defined so that:
    # ż₁ = z₂
    # ż₂ = z₃
    # ż₃ = v
    
    # Assuming we apply the same sinusoidal input through the transformation
    # v would be the appropriate transformation of u
    # For simplicity, we'll use a scaled version of the sinusoidal input
    # This is a simplification - ideally v would be computed using the full transformation
     # Find the closest time point in original_sol.t
    idx = np.argmin(np.abs(original_sol.t - t))
    x_val = np.array([original_sol.y[0, idx], original_sol.y[1, idx], original_sol.y[2, idx]])

    u_val = sinusoidal_input(t)  # Scaled for illustration purposes
    v = compute_v(x_val, u_val)
    
    dz1dt = z2
    dz2dt = z3
    dz3dt = v
    
    return np.array([dz1dt, dz2dt, dz3dt])

# Initial z-coordinates (transformed from initial x-coordinates)
z0 = np.array([
    h_func(*x0),
    Lfh_func(*x0),
    Lf2h_func(*x0)
])

# Simulate the linearized system
sol_lin = solve_ivp(
    lambda t, z: linearized_system(t, z, sol),
    t_span,
    z0,
    t_eval=t_eval,
    method='RK45'
)

# Plot the results
plt.figure(figsize=(15, 12))

# Plot original system states
plt.subplot(2, 2, 1)
plt.plot(sol.t, sol.y[0], 'r', label='x₁')
plt.plot(sol.t, sol.y[1], 'g', label='x₂')
plt.plot(sol.t, sol.y[2], 'b', label='x₃')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Original System States with Sinusoidal Input')
plt.grid(True)
plt.legend()

# Plot transformed states (z from original system)
plt.subplot(2, 2, 2)
plt.plot(sol.t, z_trajectory[0], 'r', label='z₁ = h(x)')
plt.plot(sol.t, z_trajectory[1], 'g', label='z₂ = Lfh(x)')
plt.plot(sol.t, z_trajectory[2], 'b', label='z₃ = Lf²h(x)')
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
plt.title('Linearized System States')
plt.grid(True)
plt.legend()

# Plot comparison of z₁ from both systems
plt.subplot(2, 2, 4)
plt.plot(sol.t, z_trajectory[0], 'r', label='z₁ from original system')
plt.plot(sol_lin.t, sol_lin.y[0], 'b--', label='z₁ from linearized system')
plt.xlabel('Time t')
plt.ylabel('Output')
plt.title('Comparison of Output (z₁) from Both Systems')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('simple_trajectory_comparison.png')
plt.show()

# Calculate the input for each system
u_original = np.array([sinusoidal_input(t) for t in sol.t])
v_linearized = np.array([2.0 * sinusoidal_input(t) for t in sol_lin.t])  # Scaled input for linearized system

# Plot the inputs
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(sol.t, u_original, 'k')
plt.xlabel('Time t')
plt.ylabel('Input u(t)')
plt.title('Input to Original System')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(sol_lin.t, v_linearized, 'k')
plt.xlabel('Time t')
plt.ylabel('Input v(t)')
plt.title('Input to Linearized System')
plt.grid(True)

plt.tight_layout()
plt.savefig('inputs_comparison.png')
plt.show()

# Create a more direct comparison figure
plt.figure(figsize=(15, 5))

# Compare z₁
plt.subplot(1, 3, 1)
plt.plot(sol.t, z_trajectory[0], 'r', label='From original')
plt.plot(sol_lin.t, sol_lin.y[0], 'b--', label='From linearized')
plt.xlabel('Time t')
plt.ylabel('z₁')
plt.title('Comparison of z₁')
plt.grid(True)
plt.legend()

# Compare z₂
plt.subplot(1, 3, 2)
plt.plot(sol.t, z_trajectory[1], 'r', label='From original')
plt.plot(sol_lin.t, sol_lin.y[1], 'b--', label='From linearized')
plt.xlabel('Time t')
plt.ylabel('z₂')
plt.title('Comparison of z₂')
plt.grid(True)
plt.legend()

# Compare z₃
plt.subplot(1, 3, 3)
plt.plot(sol.t, z_trajectory[2], 'r', label='From original')
plt.plot(sol_lin.t, sol_lin.y[2], 'b--', label='From linearized')
plt.xlabel('Time t')
plt.ylabel('z₃')
plt.title('Comparison of z₃')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('direct_z_comparison.png')
plt.show()

# Calculate correlation coefficients between trajectories
print("\nCorrelation Analysis:")
for i in range(3):
    corr = np.corrcoef(z_trajectory[i], sol_lin.y[i])[0, 1]
    print(f"Correlation coefficient for z{i+1}: {corr:.4f}")

# Calculate mean squared error between trajectories
mse = []
for i in range(3):
    mse_val = np.mean((z_trajectory[i] - sol_lin.y[i])**2)
    mse.append(mse_val)
    print(f"Mean squared error for z{i+1}: {mse_val:.6f}")