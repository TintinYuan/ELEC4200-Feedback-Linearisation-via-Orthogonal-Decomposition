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
print(f"LgLf²h(x) = {sp.simplify(LgLf2h)}")
print(f"Lf³h(x) = {sp.simplify(Lf3h)}")

# Define the sinusoidal input for the linearized system
def sinusoidal_v_input(t):
    """Sinusoidal control input v for the linearized system"""
    return 0.1 * np.cos(1.0 * t)

# Simulation parameters
t_span = (0, 1.5)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
x0 = np.array([1.0, 0.5, 0.2])

# Initial z-coordinates (transformed from initial x-coordinates)
z0 = np.array([
    h_func(*x0),
    Lfh_func(*x0),
    Lf2h_func(*x0)
])

# Function to map z back to x (inverse transformation)
def inverse_transform(z_values, initial_guess=None):
    """
    Find the x values that correspond to given z values using optimization.
    
    Args:
        z_values: Array of [z₁, z₂, z₃] values
        initial_guess: Initial guess for x values, defaults to [1.0, 0.5, 0.2]
    
    Returns:
        Array of [x₁, x₂, x₃] values that map to the given z values
    """
    from scipy.optimize import minimize
    
    if initial_guess is None:
        initial_guess = [1.0, 0.5, 0.2]
    
    # Function to minimize: the sum of squared differences between 
    # the desired z values and the z values computed from x
    def objective(x):
        z1_computed = h_func(*x)
        z2_computed = Lfh_func(*x)
        z3_computed = Lf2h_func(*x)
        
        error = (
            (z_values[0] - z1_computed)**2 + 
            (z_values[1] - z2_computed)**2 + 
            (z_values[2] - z3_computed)**2
        )
        return error
    
    # Perform the optimization
    result = minimize(objective, initial_guess, method='BFGS', tol=1e-6)
    
    return result.x

# Create the linearized system dynamics (direct chain of integrators)
def linearized_system(t, z):
    """
    Linearized system dynamics in z-coordinates with sinusoidal input v
    ż₁ = z₂
    ż₂ = z₃  
    ż₃ = v(t) = sinusoidal_v_input(t)
    """
    z1, z2, z3 = z
    
    v = sinusoidal_v_input(t)
    
    dz1dt = z2
    dz2dt = z3
    dz3dt = v
    
    return np.array([dz1dt, dz2dt, dz3dt])

# Simulate the linearized system with sinusoidal input v
print("Simulating linearized system...")
sol_lin = solve_ivp(
    linearized_system,
    t_span,
    z0,
    t_eval=t_eval,
    method='RK45'
)

# Now we need to compute the corresponding u(t) for the original system
# using the feedback linearization control law: u = (v - Lf³h(x)) / LgLf²h(x)
print("Computing feedback linearization control...")

# We'll solve the original system with the computed control u
# Store the control inputs
u_feedback = np.zeros_like(t_eval)
v_inputs = np.array([sinusoidal_v_input(t) for t in t_eval])

# Original system with feedback linearization control
def original_system_with_feedback(t, x):
    """
    Original system with feedback linearization control
    """
    x1_val, x2_val, x3_val = x
    
    # Get the current v input
    v_val = sinusoidal_v_input(t)
    
    # Compute Lie derivatives at current state
    Lf3h_val = Lf3h_func(x1_val, x2_val, x3_val)
    LgLf2h_val = LgLf2h_func(x1_val, x2_val, x3_val)
    
    # Compute feedback linearization control
    # u = (v - Lf³h(x)) / LgLf²h(x)
    if abs(LgLf2h_val) > 1e-10:  # Avoid division by zero
        u_val = (v_val - Lf3h_val) / LgLf2h_val
    else:
        u_val = 0  # Fallback if LgLf²h is near zero
    
    # Original system dynamics
    dx1dt = -x1_val + u_val
    dx2dt = -2*x2_val - x1_val*x3_val
    dx3dt = 3*x1_val*x2_val
    
    return np.array([dx1dt, dx2dt, dx3dt])

# Simulate the original system with feedback linearization
print("Simulating original system with feedback linearization...")
sol_feedback = solve_ivp(
    original_system_with_feedback,
    t_span,
    x0,
    t_eval=t_eval,
    method='RK45'
)

# Compute the actual u values used during simulation
for i, t in enumerate(t_eval):
    x_val = sol_feedback.y[:, i]
    v_val = sinusoidal_v_input(t)
    Lf3h_val = Lf3h_func(*x_val)
    LgLf2h_val = LgLf2h_func(*x_val)
    
    if abs(LgLf2h_val) > 1e-10:
        u_feedback[i] = (v_val - Lf3h_val) / LgLf2h_val
    else:
        u_feedback[i] = 0

# Transform the feedback linearized trajectory to z-coordinates for comparison
z_feedback = np.zeros((3, len(sol_feedback.t)))
for i in range(len(sol_feedback.t)):
    x_val = sol_feedback.y[:, i]
    z_feedback[0, i] = h_func(*x_val)
    z_feedback[1, i] = Lfh_func(*x_val)
    z_feedback[2, i] = Lf2h_func(*x_val)

# Map linearized states back to original states for comparison
# This can be computationally expensive, so we'll sample fewer points
sample_indices = np.linspace(0, len(sol_lin.t) - 1, 100, dtype=int)
x_reconstructed = np.zeros((3, len(sample_indices)))

print("Mapping linearized states back to original states (this may take a while)...")
last_x_guess = x0  # Start with the initial state as our first guess

for i, idx in enumerate(sample_indices):
    z_vals = [sol_lin.y[0, idx], sol_lin.y[1, idx], sol_lin.y[2, idx]]
    
    # Use the previous solution as the initial guess to speed up convergence
    x_reconstructed[:, i] = inverse_transform(z_vals, last_x_guess)
    last_x_guess = x_reconstructed[:, i]
    
    # Print progress
    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(sample_indices)} points")

# Plot the results
plt.figure(figsize=(15, 12))

# Plot original system with feedback linearization
plt.subplot(2, 2, 1)
plt.plot(sol_feedback.t, sol_feedback.y[0], 'r', label='x₁ (feedback)')
plt.plot(sol_feedback.t, sol_feedback.y[1], 'g', label='x₂ (feedback)')
plt.plot(sol_feedback.t, sol_feedback.y[2], 'b', label='x₃ (feedback)')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Original System with Feedback Linearization')
plt.grid(True)
plt.legend()

# Plot linearized system states
plt.subplot(2, 2, 2)
plt.plot(sol_lin.t, sol_lin.y[0], 'r', label='z₁')
plt.plot(sol_lin.t, sol_lin.y[1], 'g', label='z₂')
plt.plot(sol_lin.t, sol_lin.y[2], 'b', label='z₃')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Linearized System States')
plt.grid(True)
plt.legend()

# Plot reconstructed states from linearized system
plt.subplot(2, 2, 3)
sampled_t = sol_lin.t[sample_indices]
plt.plot(sampled_t, x_reconstructed[0], 'r--', label='x₁ reconstructed')
plt.plot(sampled_t, x_reconstructed[1], 'g--', label='x₂ reconstructed')
plt.plot(sampled_t, x_reconstructed[2], 'b--', label='x₃ reconstructed')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Reconstructed Original States from Linearized System')
plt.grid(True)
plt.legend()

# Plot comparison of feedback linearized states vs reconstructed states
plt.subplot(2, 2, 4)
# Sample the feedback linearized states at the same time points for fair comparison
sampled_feedback_indices = np.searchsorted(sol_feedback.t, sampled_t)
plt.plot(sampled_t, sol_feedback.y[0, sampled_feedback_indices], 'r', label='Feedback x₁')
plt.plot(sampled_t, x_reconstructed[0], 'r--', label='Reconstructed x₁')
plt.plot(sampled_t, sol_feedback.y[1, sampled_feedback_indices], 'g', label='Feedback x₂')
plt.plot(sampled_t, x_reconstructed[1], 'g--', label='Reconstructed x₂')
plt.plot(sampled_t, sol_feedback.y[2, sampled_feedback_indices], 'b', label='Feedback x₃')
plt.plot(sampled_t, x_reconstructed[2], 'b--', label='Reconstructed x₃')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('Comparison: Feedback Linearized vs Reconstructed States')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('feedback_linearization_comparison.png')
plt.show()

# Plot the z-coordinate comparison
plt.figure(figsize=(15, 5))

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(sol_lin.t, sol_lin.y[i], 'b-', label=f'Direct z₁₊{i} (linearized)')
    plt.plot(sol_feedback.t, z_feedback[i], 'r--', label=f'z₁₊{i} from feedback')
    plt.xlabel('Time t')
    plt.ylabel(f'z₁₊{i}')
    plt.title(f'Z-coordinate z₁₊{i} Comparison')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('direct_z_comparison.png')
plt.show()

# Plot the inputs for each system
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t_eval, u_feedback, 'k')
plt.xlabel('Time t')
plt.ylabel('Input u(t)')
plt.title('Feedback Linearization Control u(t)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_eval, v_inputs, 'k')
plt.xlabel('Time t')
plt.ylabel('Input v(t)')
plt.title('Linearized System Input v(t) (Sinusoidal)')
plt.grid(True)

plt.tight_layout()
plt.savefig('inputs_comparison.png')
plt.show()

# Plot reconstruction error over time
plt.figure(figsize=(10, 6))

for i in range(3):
    error = np.abs(sol_feedback.y[i, sampled_feedback_indices] - x_reconstructed[i])
    plt.semilogy(sampled_t, error, label=f'x{i+1} error')

plt.xlabel('Time t')
plt.ylabel('Absolute Error (log scale)')
plt.title('Reconstruction Error Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('reconstruction_error.png')
plt.show()

# Calculate correlation coefficients between z-coordinates
print("\nZ-coordinate Correlation Analysis:")
for i in range(3):
    corr = np.corrcoef(z_feedback[i], sol_lin.y[i])[0, 1]
    print(f"Correlation coefficient for z{i+1}: {corr:.6f}")

# Calculate mean squared error between z-coordinates
print("\nZ-coordinate Mean Squared Error:")
for i in range(3):
    mse_val = np.mean((z_feedback[i] - sol_lin.y[i])**2)
    print(f"Mean squared error for z{i+1}: {mse_val:.8f}")

# Calculate correlation coefficients between feedback linearized and reconstructed states
print("\nState Reconstruction Correlation Analysis:")
for i in range(3):
    corr = np.corrcoef(sol_feedback.y[i, sampled_feedback_indices], x_reconstructed[i])[0, 1]
    print(f"Correlation coefficient for x{i+1}: {corr:.6f}")

# Calculate mean squared error between feedback linearized and reconstructed states
print("\nState Reconstruction Mean Squared Error:")
for i in range(3):
    mse_val = np.mean((sol_feedback.y[i, sampled_feedback_indices] - x_reconstructed[i])**2)
    print(f"Mean squared error for x{i+1}: {mse_val:.8f}")

print(f"\nFeedback linearization successfully implemented!")
print(f"The sinusoidal input v(t) = 0.1*cos(t) was applied to the linearized system,")
print(f"and the corresponding control u(t) = (v - Lf³h(x))/LgLf²h(x) was computed for the original system.")
print(f"The z-coordinates from both approaches should be nearly identical if the feedback linearization is working correctly.")