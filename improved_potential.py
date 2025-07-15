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

# The field has a specific structure that suggests a potential of the form:
# Phi(x2, x3) = f(x2, x3) / (9*x2**2 + x3**2)
# This is based on the structure of the F2 and F3 terms

def simple_approximation():
    """
    Create a simplified form based on the structure of the vector field
    """
    # Looking at F2 and F3, we can see they have a common term in the numerator
    # and the same denominator. This suggests a potential function of the form:
    # Phi(x2, x3) = g(x2, x3) / (9*x2**2 + x3**2)
    
    # The numerator should give us F2 and F3 when differentiated
    # From observation of the structure, a good candidate is:
    k = sp.symbols('k')
    potential = k * (0.993819554534266*x2**2 + 0.110424398025967*x3**2) / sp.sqrt(9*x2**2 + x3**2)
    
    # Calculate the gradient
    grad_potential = [
        sp.diff(potential, x1),  # Should be 0
        sp.diff(potential, x2),
        sp.diff(potential, x3)
    ]
    
    # Find the value of k that minimizes error
    F2_func = sp.lambdify([x1, x2, x3], F2, "numpy")
    F3_func = sp.lambdify([x1, x2, x3], F3, "numpy")
    grad2_func = sp.lambdify([x1, x2, x3, k], grad_potential[1], "numpy")
    grad3_func = sp.lambdify([x1, x2, x3, k], grad_potential[2], "numpy")
    
    # Sample points
    sample_points = np.array([
        [0, 1, 1],
        [0, 2, 1],
        [0, 1, 2],
        [0, 0.5, 0.5],
        [0, -1, 1],
        [0, 1, -1]
    ])
    
    def objective(k_val):
        error = 0
        for point in sample_points:
            try:
                F2_val = F2_func(*point)
                F3_val = F3_func(*point)
                grad2_val = grad2_func(*point, k_val[0])
                grad3_val = grad3_func(*point, k_val[0])
                error += (F2_val - grad2_val)**2 + (F3_val - grad3_val)**2
            except:
                error += 1e6
        return error
    
    # Find optimal k
    result = minimize(objective, [1.0], method='Nelder-Mead')
    k_optimal = result.x[0]
    
    print(f"Optimal k value: {k_optimal}")
    print(f"Final error: {result.fun}")
    
    # Substitute the optimal value
    final_potential = potential.subs(k, k_optimal)
    
    return final_potential

def structure_based_approximation():
    """
    Create a more sophisticated form based on the structure of the field
    """
    # Based on examining F2 and F3, we see they have common form
    # F2 = x2 * h(x2, x3) / (9*x2**2 + x3**2)
    # F3 = x3 * h(x2, x3) / (9*x2**2 + x3**2)
    # where h(x2, x3) = constant * (0.993819554534266*x2**2 + 0.110424398025967*x3**2)
    
    # This suggests a potential of the form:
    # Phi(x2, x3) = k * log(sqrt(9*x2**2 + x3**2)) * (0.993819554534266*x2**2 + 0.110424398025967*x3**2)
    k = sp.symbols('k')
    potential_candidates = [
        # Candidate 1: Based on logarithmic form suggested by the field structure
        k * sp.log(sp.sqrt(9*x2**2 + x3**2)) * (0.993819554534266*x2**2 + 0.110424398025967*x3**2),
        
        # Candidate 2: Direct multiple of the common term
        k * (0.993819554534266*x2**2 + 0.110424398025967*x3**2),
        
        # Candidate 3: Square root form
        k * sp.sqrt(9*x2**2 + x3**2) * (0.993819554534266*x2**2 + 0.110424398025967*x3**2),
        
        # Candidate 4: Rational form
        k * (0.993819554534266*x2**2 + 0.110424398025967*x3**2) / sp.sqrt(9*x2**2 + x3**2)
    ]
    
    best_potential = None
    best_error = float('inf')
    best_k = None
    
    for i, potential in enumerate(potential_candidates):
        # Calculate the gradient
        grad_potential = [
            sp.diff(potential, x1),  # Should be 0
            sp.diff(potential, x2),
            sp.diff(potential, x3)
        ]
        
        # Create lambdified functions
        F2_func = sp.lambdify([x1, x2, x3], F2, "numpy")
        F3_func = sp.lambdify([x1, x2, x3], F3, "numpy")
        grad2_func = sp.lambdify([x1, x2, x3, k], grad_potential[1], "numpy")
        grad3_func = sp.lambdify([x1, x2, x3, k], grad_potential[2], "numpy")
        
        # Sample points
        sample_points = np.array([
            [0, 1, 1],
            [0, 2, 1],
            [0, 1, 2],
            [0, 0.5, 0.5],
            [0, -1, 1],
            [0, 1, -1]
        ])
        
        def objective(k_val):
            error = 0
            for point in sample_points:
                try:
                    F2_val = F2_func(*point)
                    F3_val = F3_func(*point)
                    grad2_val = grad2_func(*point, k_val[0])
                    grad3_val = grad3_func(*point, k_val[0])
                    error += (F2_val - grad2_val)**2 + (F3_val - grad3_val)**2
                except:
                    error += 1e6
            return error
        
        # Find optimal k
        result = minimize(objective, [1.0], method='Nelder-Mead')
        k_optimal = result.x[0]
        
        print(f"Candidate {i+1}:")
        print(f"  Optimal k value: {k_optimal}")
        print(f"  Final error: {result.fun}")
        
        if result.fun < best_error:
            best_error = result.fun
            best_potential = potential
            best_k = k_optimal
    
    # Substitute the optimal value
    final_potential = best_potential.subs(k, best_k)
    
    return final_potential, best_error

def comprehensive_approximation():
    """
    Try a more comprehensive approximation with multiple parameters
    """
    # Define parameters
    a, b, c, d = sp.symbols('a b c d')
    
    # Try a more general form
    potential = (a * (0.993819554534266*x2**2 + 0.110424398025967*x3**2) + 
                b * sp.sqrt(9*x2**2 + x3**2) + 
                c * sp.log(9*x2**2 + x3**2) * (0.993819554534266*x2**2 + 0.110424398025967*x3**2) +
                d)
    
    # Calculate the gradient
    grad_potential = [
        sp.diff(potential, x1),  # Should be 0
        sp.diff(potential, x2),
        sp.diff(potential, x3)
    ]
    
    # Create lambdified functions
    F2_func = sp.lambdify([x1, x2, x3], F2, "numpy")
    F3_func = sp.lambdify([x1, x2, x3], F3, "numpy")
    grad2_func = sp.lambdify([x1, x2, x3, a, b, c, d], grad_potential[1], "numpy")
    grad3_func = sp.lambdify([x1, x2, x3, a, b, c, d], grad_potential[2], "numpy")
    
    # Sample points
    sample_points = np.array([
        [0, 1, 1],
        [0, 2, 1],
        [0, 1, 2],
        [0, 0.5, 0.5],
        [0, -1, 1],
        [0, 1, -1],
        [0, 2, 2],
        [0, 0.1, 0.1]
    ])
    
    def objective(params):
        error = 0
        for point in sample_points:
            try:
                F2_val = F2_func(*point)
                F3_val = F3_func(*point)
                grad2_val = grad2_func(*point, *params)
                grad3_val = grad3_func(*point, *params)
                error += (F2_val - grad2_val)**2 + (F3_val - grad3_val)**2
            except Exception as e:
                error += 1e6
        return error
    
    # Find optimal parameters
    result = minimize(objective, [0, 0, 1, 0], method='Nelder-Mead', options={'maxiter': 10000})
    optimal_params = result.x
    
    print(f"Optimal parameters: a={optimal_params[0]}, b={optimal_params[1]}, c={optimal_params[2]}, d={optimal_params[3]}")
    print(f"Final error: {result.fun}")
    
    # Substitute the optimal values
    final_potential = potential.subs({a: optimal_params[0], b: optimal_params[1], 
                                    c: optimal_params[2], d: optimal_params[3]})
    
    return final_potential, result.fun

def validate_potential(potential, F1, F2, F3, x1, x2, x3):
    """
    Validate the approximated potential by comparing its gradient with the original field
    """
    gradient = [
        sp.diff(potential, x1),
        sp.diff(potential, x2),
        sp.diff(potential, x3)
    ]
    
    print("\nGradient of potential:")
    print(f"d/dx1 = {sp.simplify(gradient[0])}")
    print(f"d/dx2 = {sp.simplify(gradient[1])}")
    print(f"d/dx3 = {sp.simplify(gradient[2])}")
    
    # Calculate error
    error = [F1 - gradient[0], F2 - gradient[1], F3 - gradient[2]]
    
    print("\nError in each component:")
    print(f"Error_x1 = {sp.simplify(error[0])}")
    print(f"Error_x2 = {sp.simplify(error[1])}")
    print(f"Error_x3 = {sp.simplify(error[2])}")
    
    # Create lambdified functions for field and gradient
    F_funcs = [sp.lambdify([x1, x2, x3], F, "numpy") for F in [F1, F2, F3]]
    grad_funcs = [sp.lambdify([x1, x2, x3], g, "numpy") for g in gradient]
    
    # Sample points
    sample_points = [
        (0, 1, 1),
        (0, 2, 1),
        (0, 1, 2),
        (0, -1, 1),
        (0, 1, -1),
        (0, 2, 2)
    ]
    
    total_error = 0
    for point in sample_points:
        point_error = 0
        for i in range(3):
            try:
                field_val = F_funcs[i](*point)
                grad_val = grad_funcs[i](*point)
                point_error += (field_val - grad_val)**2
            except Exception as e:
                print(f"Error at point {point}, component {i}: {e}")
                point_error += 1e6
        
        total_error += point_error
        print(f"Error at point {point}: {np.sqrt(point_error)}")
    
    mean_error = total_error / len(sample_points)
    print(f"\nMean squared error: {mean_error}")
    
    return mean_error

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
    pot_func = sp.lambdify([x1, x2, x3], potential, "numpy")
    
    # Create grid
    x2_vals = np.linspace(-3, 3, 30)
    x3_vals = np.linspace(-3, 3, 30)
    X2, X3 = np.meshgrid(x2_vals, x3_vals)
    
    # Evaluate field and gradient
    F2_vals = np.zeros_like(X2)
    F3_vals = np.zeros_like(X3)
    grad2_vals = np.zeros_like(X2)
    grad3_vals = np.zeros_like(X3)
    pot_vals = np.zeros_like(X2)
    
    for i in range(len(x2_vals)):
        for j in range(len(x3_vals)):
            try:
                x2_val = x2_vals[i]
                x3_val = x3_vals[j]
                F2_vals[j, i] = F2_func(0, x2_val, x3_val)
                F3_vals[j, i] = F3_func(0, x2_val, x3_val)
                grad2_vals[j, i] = grad2_func(0, x2_val, x3_val)
                grad3_vals[j, i] = grad3_func(0, x2_val, x3_val)
                pot_vals[j, i] = pot_func(0, x2_val, x3_val)
            except Exception as e:
                pass
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Plot original field
    plt.subplot(221)
    plt.quiver(X2, X3, F2_vals, F3_vals, scale=500)
    plt.title('Original Vector Field [F2, F3]')
    plt.xlabel('x2')
    plt.ylabel('x3')
    plt.grid(True)
    
    # Plot gradient of potential
    plt.subplot(222)
    plt.quiver(X2, X3, grad2_vals, grad3_vals, scale=500)
    plt.title('Gradient of Approximated Potential')
    plt.xlabel('x2')
    plt.ylabel('x3')
    plt.grid(True)
    
    # Plot error magnitude
    plt.subplot(223)
    error_mag = np.sqrt((F2_vals - grad2_vals)**2 + (F3_vals - grad3_vals)**2)
    plt.pcolormesh(X2, X3, np.log1p(error_mag), shading='auto', cmap='hot')
    plt.colorbar(label='Log Error Magnitude')
    plt.title('Log Error Magnitude')
    plt.xlabel('x2')
    plt.ylabel('x3')
    
    # Plot approximated potential
    plt.subplot(224)
    plt.contourf(X2, X3, pot_vals, 30, cmap='viridis')
    plt.colorbar(label='Potential Value')
    plt.title('Approximated Scalar Potential')
    plt.xlabel('x2')
    plt.ylabel('x3')
    
    plt.tight_layout()
    plt.savefig('potential_approximation.png', dpi=300)
    plt.close()

def main():
    # Check if the field has curl
    curl_x = sp.diff(F3, x2) - sp.diff(F2, x3)
    curl_y = sp.diff(F1, x3) - sp.diff(F3, x1)
    curl_z = sp.diff(F2, x1) - sp.diff(F1, x2)
    
    print("Curl components:")
    print(f"curl_x = {sp.simplify(curl_x)}")
    print(f"curl_y = {sp.simplify(curl_y)}")
    print(f"curl_z = {sp.simplify(curl_z)}")
    
    print("\n1. Using simple approximation")
    simple_potential = simple_approximation()
    simple_error = validate_potential(simple_potential, F1, F2, F3, x1, x2, x3)
    
    print("\n2. Using structure-based approximation")
    structure_potential, structure_error = structure_based_approximation()
    validate_potential(structure_potential, F1, F2, F3, x1, x2, x3)
    
    print("\n3. Using comprehensive approximation")
    comprehensive_potential, comp_error = comprehensive_approximation()
    validate_potential(comprehensive_potential, F1, F2, F3, x1, x2, x3)
    
    # Find the best potential
    potentials = [
        ("Simple", simple_potential, simple_error),
        ("Structure-based", structure_potential, structure_error),
        ("Comprehensive", comprehensive_potential, comp_error)
    ]
    
    best_name, best_potential, best_error = min(potentials, key=lambda x: x[2])
    
    print(f"\nBest approximation: {best_name} with error {best_error}")
    print(f"Best potential: {sp.simplify(best_potential)}")
    
    # Plot the best potential
    plot_field_vs_gradient(best_potential, F1, F2, F3, x1, x2, x3)
    
    # Save to file
    with open('potential_approximation.txt', 'w') as f:
        f.write("Vector Field Approximation Results\n")
        f.write("=================================\n\n")
        
        f.write("Original Vector Field:\n")
        f.write(f"F1 = {F1}\n")
        f.write(f"F2 = {F2}\n")
        f.write(f"F3 = {F3}\n\n")
        
        f.write("Curl Components:\n")
        f.write(f"curl_x = {sp.simplify(curl_x)}\n")
        f.write(f"curl_y = {sp.simplify(curl_y)}\n")
        f.write(f"curl_z = {sp.simplify(curl_z)}\n\n")
        
        f.write("Best Approximation Method: " + best_name + "\n")
        f.write(f"Error: {best_error}\n\n")
        
        f.write("Approximated Scalar Potential:\n")
        f.write(f"Phi = {sp.simplify(best_potential)}\n\n")
        
        # Gradient
        gradient = [sp.diff(best_potential, var) for var in [x1, x2, x3]]
        f.write("Gradient of Potential:\n")
        for i, var in enumerate([x1, x2, x3]):
            f.write(f"d/d{var} = {sp.simplify(gradient[i])}\n")
        
        # Error
        error = [F1 - gradient[0], F2 - gradient[1], F3 - gradient[2]]
        f.write("\nError in each component:\n")
        for i, var in enumerate([x1, x2, x3]):
            f.write(f"Error_{var} = {sp.simplify(error[i])}\n")
        
        f.write("\nA visualization of the field and potential has been saved as 'potential_approximation.png'")

if __name__ == "__main__":
    main()
