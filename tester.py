import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def compare_original_and_linearized_dynamics(x_initial, t_span, v_func):
    """
    Compares the original system dynamics with the linearized dynamics 
    under the same control input.
    
    Parameters:
    x_initial (list): Initial state [x1, x2, x3]
    t_span (tuple): Simulation time span (start, end)
    v_func (function): Input function v(t)
    
    Returns:
    dict: Simulation results
    """
    # Define symbolic variables
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    variables = [x1, x2, x3]
    dim = len(variables)
    
    # Define system dynamics (your provided f(x) and g(x))
    fx = sp.Matrix([-x1, -2*x2 - x1*x3, 3*x1*x2])
    gx = sp.Matrix([1, 0, 0])
    
    # Define output function h(x)
    h = -1.98794802540439*x2**2 - 0.662649372160206*x3**2
    
    # Create companion form matrices
    Ac = sp.Matrix(dim, dim, lambda i, j: 1 if j == i+1 else 0)
    Bc = sp.Matrix(dim, 1, lambda i, j: 1 if i == dim-1 else 0)
    
    # Function to compute Lie derivative
    def lie_derivative(function, vector_field, vars):
        result = 0
        for i, var in enumerate(vars):
            result += sp.diff(function, var) * vector_field[i]
        return result
    
    # Compute z-coordinates (Lie derivatives)
    z_transform = [h]
    for i in range(1, dim):
        z_transform.append(lie_derivative(z_transform[-1], fx, variables))
    
    # Compute L_g L_f^(n-1) h(x) (gamma)
    gamma = lie_derivative(z_transform[-2], gx, variables)
    
    # Compute L_f^n h(x)
    lfn_h = lie_derivative(z_transform[-2], fx, variables)
    
    # Compute alpha(x) = L_f^n h(x) / (L_g L_f^(n-1) h(x))
    alpha = lfn_h / gamma
    
    # For better numerical stability, we can simplify expressions
    z_transform = [expr.simplify() for expr in z_transform]
    gamma = gamma.simplify()
    alpha = alpha.simplify()
    
    # Create lambdified functions for z-coordinates
    z_funcs = [sp.lambdify([x1, x2, x3], expr) for expr in z_transform]
    gamma_func = sp.lambdify([x1, x2, x3], gamma)
    alpha_func = sp.lambdify([x1, x2, x3], alpha)
    
    # Initial z-coordinates
    z_initial = [func(*x_initial) for func in z_funcs]
    
    # Original system simulation
    def original_system_ode(t, x):
        x1_val, x2_val, x3_val = x
        # Calculate control input u based on v
        v_val = v_func(t)
        gamma_val = gamma_func(x1_val, x2_val, x3_val)
        alpha_val = alpha_func(x1_val, x2_val, x3_val)
        
        # u = (v - alpha) / gamma
        u_val = (v_val - alpha_val) / gamma_val
        
        # Calculate state derivatives using f(x) + g(x)u
        f_val = np.array([
            -x1_val,
            -2*x2_val - x1_val*x3_val,
            3*x1_val*x2_val
        ])
        g_val = np.array([1, 0, 0])
        
        return f_val + g_val * u_val
    
    # Linearized system simulation
    def linearized_system_ode(t, z):
        v_val = v_func(t)
        # z_dot = Ac*z + Bc*v
        Ac_np = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        Bc_np = np.array([0, 0, 1]).reshape(3, 1)
        
        return (Ac_np @ z.reshape(-1, 1) + Bc_np * v_val).flatten()
    
    # Solve ODEs
    solution_original = solve_ivp(
        original_system_ode, t_span, x_initial, 
        method='RK45', dense_output=True, rtol=1e-6, atol=1e-9
    )
    
    solution_linearized = solve_ivp(
        linearized_system_ode, t_span, z_initial, 
        method='RK45', dense_output=True, rtol=1e-6, atol=1e-9
    )
    
    # Calculate z-coordinates from original system for comparison
    t_points = solution_original.t
    x_trajectory = solution_original.y
    z_from_x = np.zeros((dim, len(t_points)))
    
    for i, t in enumerate(t_points):
        x_at_t = x_trajectory[:, i]
        for j in range(dim):
            z_from_x[j, i] = z_funcs[j](*x_at_t)
    
    # Store results
    results = {
        'original_system': {
            't': solution_original.t,
            'x_states': solution_original.y,
            'z_from_x': z_from_x
        },
        'linearized_system': {
            't': solution_linearized.t,
            'z_states': solution_linearized.y
        },
        'transformations': {
            'z_transform_symbolic': z_transform,
            'gamma_symbolic': gamma,
            'alpha_symbolic': alpha
        }
    }
    
    return results

def plot_comparison(results):
    """
    Plots the comparison between original and linearized dynamics
    """
    plt.figure(figsize=(15, 12))
    
    # Plot original state variables
    plt.subplot(3, 2, 1)
    plt.plot(results['original_system']['t'], results['original_system']['x_states'][0], 'b-', label='x₁')
    plt.plot(results['original_system']['t'], results['original_system']['x_states'][1], 'r-', label='x₂')
    plt.plot(results['original_system']['t'], results['original_system']['x_states'][2], 'g-', label='x₃')
    plt.grid(True)
    plt.legend()
    plt.title('Original State Variables')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Compare z₁ from both systems
    plt.subplot(3, 2, 2)
    plt.plot(results['original_system']['t'], results['original_system']['z_from_x'][0], 'b-', label='z₁ from original')
    plt.plot(results['linearized_system']['t'], results['linearized_system']['z_states'][0], 'r--', label='z₁ linearized')
    plt.grid(True)
    plt.legend()
    plt.title('z₁ Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Compare z₂ from both systems
    plt.subplot(3, 2, 3)
    plt.plot(results['original_system']['t'], results['original_system']['z_from_x'][1], 'b-', label='z₂ from original')
    plt.plot(results['linearized_system']['t'], results['linearized_system']['z_states'][1], 'r--', label='z₂ linearized')
    plt.grid(True)
    plt.legend()
    plt.title('z₂ Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Compare z₃ from both systems
    plt.subplot(3, 2, 4)
    plt.plot(results['original_system']['t'], results['original_system']['z_from_x'][2], 'b-', label='z₃ from original')
    plt.plot(results['linearized_system']['t'], results['linearized_system']['z_states'][2], 'r--', label='z₃ linearized')
    plt.grid(True)
    plt.legend()
    plt.title('z₃ Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Calculate and plot errors
    plt.subplot(3, 2, 5)
    # Interpolate linearized z values to match original time points
    from scipy.interpolate import interp1d
    
    t_original = results['original_system']['t']
    t_linear = results['linearized_system']['t']
    z_linear = results['linearized_system']['z_states']
    
    errors = np.zeros((3, len(t_original)))
    
    for i in range(3):
        interpolator = interp1d(t_linear, z_linear[i], kind='cubic', bounds_error=False, fill_value="extrapolate")
        z_linear_interp = interpolator(t_original)
        errors[i] = results['original_system']['z_from_x'][i] - z_linear_interp
    
    plt.plot(t_original, errors[0], 'b-', label='z₁ error')
    plt.plot(t_original, errors[1], 'r-', label='z₂ error')
    plt.plot(t_original, errors[2], 'g-', label='z₃ error')
    plt.grid(True)
    plt.legend()
    plt.title('Error between original and linearized z')
    plt.xlabel('Time')
    plt.ylabel('Error')
    
    # Plot the control input v(t)
    plt.subplot(3, 2, 6)
    t_plot = np.linspace(t_original[0], t_original[-1], 1000)
    v_plot = np.array([v_func(t) for t in t_plot])
    plt.plot(t_plot, v_plot, 'k-')
    plt.grid(True)
    plt.title('Control Input v(t)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()

# Define a control input function
def v_func(t):
    # Example: sinusoidal input
    return np.sin(t)
    
    # Alternative inputs:
    # Step input:
    # return 1.0 if t >= 1.0 else 0.0
    
    # Ramp input:
    # return t if t < 5.0 else 5.0

# Run the simulation and plot results
x_initial = [0.5, 0.2, 0.3]  # Initial state
t_span = (0, 10)  # Simulation time span

# Run the comparison
results = compare_original_and_linearized_dynamics(x_initial, t_span, v_func)

# Plot the results
plot_comparison(results)

# Print the symbolic expressions for the z-coordinates
print("Symbolic z-coordinates:")
for i, expr in enumerate(results['transformations']['z_transform_symbolic']):
    print(f"z_{i+1} = {expr}")

print("\nGamma (gain term):")
print(results['transformations']['gamma_symbolic'])

print("\nAlpha (compensation term):")
print(results['transformations']['alpha_symbolic'])