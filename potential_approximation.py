import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define your variables
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Define the field components
F1 = 0
F2 = 35.9987702689914*x2*(0.993819554534266*x2**2 + 0.110424398025967*x3**2)/(9*x2**2 + x3**2)
F3 = 11.9995900896638*x3*(0.993819554534266*x2**2 + 0.110424398025967*x3**2)/(9*x2**2 + x3**2)

def check_curl_free(F1, F2, F3, x1, x2, x3):
    """
    Check if the vector field is curl-free (conservative)
    """
    curl_x = sp.diff(F3, x2) - sp.diff(F2, x3)
    curl_y = sp.diff(F1, x3) - sp.diff(F3, x1)
    curl_z = sp.diff(F2, x1) - sp.diff(F1, x2)
    
    print("Curl components:")
    print(f"Curl_x = {sp.simplify(curl_x)}")
    print(f"Curl_y = {sp.simplify(curl_y)}")
    print(f"Curl_z = {sp.simplify(curl_z)}")
    
    # Create lambdified functions to evaluate curl magnitude at sample points
    curl_funcs = [sp.lambdify([x1, x2, x3], c, "numpy") for c in [curl_x, curl_y, curl_z]]
    
    # Sample points to evaluate curl magnitude
    points = np.array([
        [0, 1, 1],
        [0, 2, 1],
        [0, 1, 2],
        [0, -1, 1],
        [0, 1, -1]
    ])
    
    max_curl_mag = 0
    for p in points:
        curl_vals = [abs(f(*p)) for f in curl_funcs]
        curl_mag = np.sqrt(sum(c**2 for c in curl_vals))
        max_curl_mag = max(max_curl_mag, curl_mag)
    
    print(f"Maximum curl magnitude at sample points: {max_curl_mag}")
    print(f"Field is {'approximately' if max_curl_mag < 1e-6 else 'not'} curl-free.\n")
    
    return max_curl_mag < 1e-6

def numerical_potential_approximation(F1, F2, F3, x1, x2, x3):
    """
    Approximate scalar potential using parametric rational function and optimization
    """
    print("Using numerical approximation for scalar potential...")
    
    # Define a parametric form for the potential
    # Based on the structure of your vector field, using a rational form
    params = sp.symbols('a:15')  # Define parameters
    
    # Create a rational potential function - numerator and denominator
    numerator = (
        params[0] + 
        params[1]*x2 + params[2]*x3 +
        params[3]*x2**2 + params[4]*x3**2 + params[5]*x2*x3 +
        params[6]*x2**3 + params[7]*x3**3 + params[8]*x2**2*x3 + params[9]*x2*x3**2
    )
    
    denominator = (
        1 + params[10]*(x2**2) + params[11]*(x3**2) + params[12]*(x2*x3) +
        params[13]*(x2**2*x3**2) + params[14]*(9*x2**2 + x3**2)
    )
    
    potential = numerator / denominator
    
    # Compute the gradient of this potential
    grad_potential = [
        sp.diff(potential, x1),  # Should be 0 since no x1 dependence
        sp.diff(potential, x2),
        sp.diff(potential, x3)
    ]
    
    # Create functions to evaluate the field components
    F_funcs = [sp.lambdify([x1, x2, x3], F, "numpy") for F in [F1, F2, F3]]
    
    # Create functions that evaluate gradient components with given parameters
    grad_funcs = [sp.lambdify([x1, x2, x3] + list(params), g, "numpy") for g in grad_potential]
    
    # Sample points where we want our potential's gradient to match the field
    # Avoid points where denominator might be close to zero
    sample_points = np.array([
        [0, 1, 0.5],
        [0, 2, 1],
        [0, 0.5, 1.5],
        [0, 1.5, 2],
        [0, 2, 2],
        [0, -1, 0.5],
        [0, -2, 1],
        [0, -0.5, 1.5],
        [0, 1, -0.5],
        [0, 0.5, -1],
        [0, 0.1, 0.1],
        [0, 0.2, 0.2]
    ])
    
    # Weight certain points more heavily (especially near the origin)
    weights = np.ones(len(sample_points))
    for i, point in enumerate(sample_points):
        if np.abs(point[1]) < 0.3 and np.abs(point[2]) < 0.3:  # Close to origin
            weights[i] = 5.0
    
    # Objective function to minimize
    def objective(p):
        error = 0
        for i, point in enumerate(sample_points):
            weight = weights[i]
            for j in range(3):
                field_val = F_funcs[j](*point)
                grad_val = grad_funcs[j](*point, *p)
                try:
                    error += weight * (field_val - grad_val)**2
                except:
                    error += 1e6  # Large penalty for numerical issues
        return error
    
    # Multiple optimization attempts with different initial guesses
    best_params = None
    best_error = float('inf')
    
    # Initial guesses to try
    initial_guesses = [
        np.zeros(15),  # All zeros
        np.random.randn(15) * 0.1,  # Small random values
        np.random.randn(15),  # Larger random values
        np.ones(15) * 0.1  # Small positive values
    ]
    
    for init_guess in initial_guesses:
        result = minimize(objective, init_guess, method='Nelder-Mead', 
                         options={'maxiter': 10000})
        
        if result.fun < best_error:
            best_error = result.fun
            best_params = result.x
            
    print(f"Optimization result: success={result.success}, final error={best_error}")
    
    # Substitute optimized parameters back into the potential
    optimized_potential = potential.subs({params[i]: best_params[i] for i in range(len(params))})
    
    return optimized_potential, best_params

def validate_potential(potential, F1, F2, F3, x1, x2, x3):
    """
    Validate the approximated potential by comparing its gradient with the original field
    """
    gradient = [
        sp.diff(potential, x1),
        sp.diff(potential, x2),
        sp.diff(potential, x3)
    ]
    
    # Create lambdified functions for field and gradient
    F_funcs = [sp.lambdify([x1, x2, x3], F, "numpy") for F in [F1, F2, F3]]
    grad_funcs = [sp.lambdify([x1, x2, x3], g, "numpy") for g in gradient]
    
    # Sample points
    x2_vals = np.linspace(-3, 3, 10)
    x3_vals = np.linspace(-3, 3, 10)
    X2, X3 = np.meshgrid(x2_vals, x3_vals)
    
    # Calculate errors
    total_error = 0
    point_count = 0
    
    for i in range(len(x2_vals)):
        for j in range(len(x3_vals)):
            x2_val = x2_vals[i]
            x3_val = x3_vals[j]
            
            # Skip points close to origin where denominator might be small
            if abs(x2_val) < 0.1 and abs(x3_val) < 0.1:
                continue
                
            point = (0, x2_val, x3_val)  # x1 is always 0 for this field
            
            error_squared = 0
            for k in range(3):
                field_val = F_funcs[k](*point)
                grad_val = grad_funcs[k](*point)
                try:
                    error_squared += (field_val - grad_val)**2
                except:
                    # Skip points with numerical issues
                    error_squared = float('inf')
                    break
            
            if error_squared < float('inf'):
                total_error += error_squared
                point_count += 1
    
    mean_squared_error = total_error / point_count if point_count > 0 else float('inf')
    print(f"\nValidation MSE: {mean_squared_error}")
    
    return mean_squared_error

def plot_field_vs_gradient(potential, F1, F2, F3, x1, x2, x3):
    """
    Plot the original field vs gradient of the approximated potential
    """
    gradient = [
        sp.diff(potential, x1),
        sp.diff(potential, x2),
        sp.diff(potential, x3)
    ]
    
    # Create lambdified functions
    F2_func = sp.lambdify([x1, x2, x3], F2, "numpy")
    F3_func = sp.lambdify([x1, x2, x3], F3, "numpy")
    grad2_func = sp.lambdify([x1, x2, x3], gradient[1], "numpy")
    grad3_func = sp.lambdify([x1, x2, x3], gradient[2], "numpy")
    
    # Create grid
    x2_vals = np.linspace(-3, 3, 20)
    x3_vals = np.linspace(-3, 3, 20)
    X2, X3 = np.meshgrid(x2_vals, x3_vals)
    
    # Evaluate field and gradient
    F2_vals = np.zeros_like(X2)
    F3_vals = np.zeros_like(X3)
    grad2_vals = np.zeros_like(X2)
    grad3_vals = np.zeros_like(X3)
    
    for i in range(len(x2_vals)):
        for j in range(len(x3_vals)):
            try:
                F2_vals[j, i] = F2_func(0, x2_vals[i], x3_vals[j])
                F3_vals[j, i] = F3_func(0, x2_vals[i], x3_vals[j])
                grad2_vals[j, i] = grad2_func(0, x2_vals[i], x3_vals[j])
                grad3_vals[j, i] = grad3_func(0, x2_vals[i], x3_vals[j])
            except:
                F2_vals[j, i] = 0
                F3_vals[j, i] = 0
                grad2_vals[j, i] = 0
                grad3_vals[j, i] = 0
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot original field
    plt.subplot(221)
    plt.quiver(X2, X3, F2_vals, F3_vals)
    plt.title('Original Vector Field [F2, F3]')
    plt.xlabel('x2')
    plt.ylabel('x3')
    plt.grid(True)
    
    # Plot gradient of potential
    plt.subplot(222)
    plt.quiver(X2, X3, grad2_vals, grad3_vals)
    plt.title('Gradient of Approximated Potential')
    plt.xlabel('x2')
    plt.ylabel('x3')
    plt.grid(True)
    
    # Plot error magnitude
    plt.subplot(223)
    error_mag = np.sqrt((F2_vals - grad2_vals)**2 + (F3_vals - grad3_vals)**2)
    plt.pcolormesh(X2, X3, error_mag, shading='auto', cmap='hot')
    plt.colorbar(label='Error Magnitude')
    plt.title('Error Magnitude')
    plt.xlabel('x2')
    plt.ylabel('x3')
    
    # Plot approximated potential
    pot_func = sp.lambdify([x1, x2, x3], potential, "numpy")
    pot_vals = np.zeros_like(X2)
    
    for i in range(len(x2_vals)):
        for j in range(len(x3_vals)):
            try:
                pot_vals[j, i] = pot_func(0, x2_vals[i], x3_vals[j])
            except:
                pot_vals[j, i] = 0
    
    plt.subplot(224)
    plt.contourf(X2, X3, pot_vals, 20, cmap='viridis')
    plt.colorbar(label='Potential Value')
    plt.title('Approximated Scalar Potential')
    plt.xlabel('x2')
    plt.ylabel('x3')
    
    plt.tight_layout()
    plt.savefig('potential_approximation.png', dpi=300)
    plt.show()

def main():
    # First check if the field is curl-free
    is_curl_free = check_curl_free(F1, F2, F3, x1, x2, x3)
    
    if not is_curl_free:
        print("Warning: Field is not perfectly curl-free, but we'll approximate a potential anyway.")
    
    # Approximate the scalar potential
    potential, params = numerical_potential_approximation(F1, F2, F3, x1, x2, x3)
    
    print("\nApproximated scalar potential:")
    print(sp.simplify(potential))
    
    # Validate the potential
    validate_potential(potential, F1, F2, F3, x1, x2, x3)
    
    # Plot the field vs gradient
    plot_field_vs_gradient(potential, F1, F2, F3, x1, x2, x3)
    
    # Save results to file
    with open('potential_approximation.txt', 'w') as f:
        f.write("Approximated Scalar Potential:\n")
        f.write(f"{sp.simplify(potential)}\n\n")
        
        f.write("Gradient of Potential:\n")
        for i, var in enumerate([x1, x2, x3]):
            gradient = sp.diff(potential, var)
            f.write(f"d/d{var} = {sp.simplify(gradient)}\n")
        
        f.write("\nOptimized Parameters:\n")
        for i, p in enumerate(params):
            f.write(f"a{i} = {p}\n")
        
        # Calculate the error between field and gradient
        gradient = [sp.diff(potential, var) for var in [x1, x2, x3]]
        error = [F1 - gradient[0], F2 - gradient[1], F3 - gradient[2]]
        
        f.write("\nError in each component:\n")
        for i, var in enumerate([x1, x2, x3]):
            f.write(f"Error_{var} = {sp.simplify(error[i])}\n")

if __name__ == "__main__":
    main()
