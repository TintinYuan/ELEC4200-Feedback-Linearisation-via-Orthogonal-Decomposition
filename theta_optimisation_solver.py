import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def objective(theta):
    """
    Objective function to minimize - we'll minimize the sum of squared constraint violations
    """
    return 0  # Since we're only concerned with finding a feasible solution

def constraint_violations(theta):
    """
    Calculates the constraint violations as a vector
    """
    theta0, theta1, theta2, theta3, theta4, theta5 = theta
    
    # Norm constraints
    c_norm1 = theta0**2 + theta1**2 + theta2**2 - 1
    c_norm2 = theta3**2 + theta4**2 + theta5**2 - 1
    
    # Original constraints C1-C9
    c1 = theta0 * theta3
    c2 = theta2 * theta3
    c3 = (- 108 * theta0 * theta3 + 36 * theta0 * theta3 + 
          324 * theta0 * theta4 - 972 * theta1 * theta3 - 324 * theta2 * theta5)
    c4 = (- 216 * theta2 * theta3 + 72 * theta0 * theta5 - 
          648 * theta1 * theta5)
    c5 = (12 * theta0 * theta3 + 108 * theta0 * theta4 - 324 * theta1 * theta3 - 
          108 * theta2 * theta5 + 108 * theta0 * theta4 - 36 * theta1 * theta3 - 
          324 * theta1 * theta4 + 36 * theta2 * theta5)
    c6 = (24 * theta0 * theta5 - 216 * theta1 * theta5 + 
          72 * theta2 * theta4)
    c7 = 36 * theta0 * theta4 - 12 * theta1 * theta3 - 108 * theta1 * theta4 + 12 * theta2 * theta5 + 36 * theta1 * theta4
    c8 = theta2 * theta4
    c9 = theta1 * theta4
    
    return np.array([c_norm1, c_norm2, c1, c2, c3, c4, c5, c6, c7, c8, c9])

def total_violation(theta):
    """
    Returns the sum of squared constraint violations
    """
    violations = constraint_violations(theta)
    return np.sum(violations**2)

def solve_with_initial_point(initial_guess):
    """
    Attempt to solve the optimization problem with a given initial guess
    """
    result = minimize(
        total_violation,
        initial_guess,
        method='SLSQP',
        options={'ftol': 1e-10, 'disp': False, 'maxiter': 1000}
    )
    
    return result

def random_unit_vector(size=3):
    """
    Generate a random unit vector of specified size
    """
    vec = np.random.randn(size)
    return vec / np.linalg.norm(vec)

def verify_solution(theta):
    """
    Verify that a solution satisfies all constraints
    """
    violations = constraint_violations(theta)
    max_violation = np.max(np.abs(violations))
    
    print(f"Max constraint violation: {max_violation:.6e}")
    
    if max_violation < 1e-6:
        print("All constraints are satisfied!")
    else:
        print("Constraints are not satisfied.")
        for i, v in enumerate(violations):
            print(f"Constraint {i}: {v:.6e}")

def visualize_solutions(solutions):
    """
    Visualize the first three components and last three components of solutions
    """
    if not solutions:
        print("No solutions to visualize")
        return
        
    fig = plt.figure(figsize=(15, 6))
    
    # First three components
    ax1 = fig.add_subplot(121, projection='3d')
    for sol in solutions:
        ax1.scatter(sol[0], sol[1], sol[2], marker='o')
    
    # Add a unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='b', alpha=0.1)
    
    ax1.set_xlabel('θ0')
    ax1.set_ylabel('θ1')
    ax1.set_zlabel('θ2')
    ax1.set_title('First three components')
    
    # Last three components
    ax2 = fig.add_subplot(122, projection='3d')
    for sol in solutions:
        ax2.scatter(sol[3], sol[4], sol[5], marker='o')
    
    # Add a unit sphere
    ax2.plot_surface(x, y, z, color='b', alpha=0.1)
    
    ax2.set_xlabel('θ3')
    ax2.set_ylabel('θ4')
    ax2.set_zlabel('θ5')
    ax2.set_title('Last three components')
    
    plt.tight_layout()
    plt.savefig('optimization_solutions.png')
    plt.close()
    print("Visualization saved as 'optimization_solutions.png'")

def main():
    """
    Main function to solve the optimization problem with multiple initial guesses
    """
    # Number of random starting points
    n_starts = 100
    tolerance = 1e-6
    solutions = []
    
    # Known solutions from analytical analysis
    # Case 1: θ3 = 0
    known_solutions = [
        # Solution family 1: [0, 0, 0, 0, cos(α), sin(α)]
        np.array([0.9939, 0.1104, 0, 0, 0, -1]),  # α = 0
        np.array([-0.9939, -0.1104, 0, 0, 0, 1]),  # α = π/2
    ]
    
    # BUG delete known solution
    # Check known solutions first
    print("Verifying analytically derived solutions:")
    for i, sol in enumerate(known_solutions):
        print(f"\nKnown Solution {i+1}:")
        print(sol)
        verify_solution(sol)
        if np.max(np.abs(constraint_violations(sol))) < tolerance:
            solutions.append(sol)
    
    # Random search with multiple starting points
    print("\nSearching for solutions using numerical optimisation:")
    
    for i in range(n_starts):
        # Random initial guess with unit norm constraints
        first_half = random_unit_vector()
        second_half = random_unit_vector()
        initial_guess = np.concatenate([first_half, second_half])
        
        # Run optimization
        result = solve_with_initial_point(initial_guess)
        
        # Check if the solution is valid
        if result.success and total_violation(result.x) < tolerance:
            # Check if this solution is already found (within some tolerance)
            is_new = True
            for sol in solutions:
                if np.linalg.norm(result.x - sol) < 0.01:
                    is_new = False
                    break
            
            if is_new:
                print(f"\nFound new solution (iteration {i+1}):")
                print(result.x)
                verify_solution(result.x)
                solutions.append(result.x)
    
    print(f"\nTotal unique solutions found: {len(solutions)}")
    
    # Visualize all found solutions
    visualize_solutions(solutions)
    
    # Print all solutions in a format easy to copy
    print("\nAll solutions found:")
    for i, sol in enumerate(solutions):
        print(f"Solution {i+1}:")
        print("[" + ", ".join([f"{x:.10f}" for x in sol]) + "]")

if __name__ == "__main__":
    main()