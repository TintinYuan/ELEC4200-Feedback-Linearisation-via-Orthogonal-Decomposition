# G_BH solver with uncancelled denominator

import sympy as sp
import numpy as np
from scipy.optimize import minimize 
from utils import gram_schmidt2, lie_bracket2, func_chooser2, constraint_violations2, verify_solution2, symbolic_integration, is_curl_free
import UncanceledRational as ur
from theta_optimisation_solver import random_unit_vector

x1, x2, x3 = sp.symbols('x1 x2 x3')
variable_x = sp.Matrix([x1, x2, x3])

fx, gx = func_chooser2(2) # with x1, x2, x3 defined inside the func_chooser

# Step 1: Compute ad_f g = [f, g]
ad_f_g = lie_bracket2(fx, gx, variable_x)
# ad_f_g = sp.simplify(ad_f_g)
print("ad_f g = [f, g] =")
print(ad_f_g)
print()

# Step 2: Compute ad_f² g = ad_f(ad_f g) = [f, [f, g]]
ad_f2_g = lie_bracket2(fx, ad_f_g, variable_x)
# ad_f2_g = sp.sympify(ad_f2_g)
print("ad_f² g = [f, [f, g]] =")
print(ad_f2_g)
print()

vector = [gx, ad_f_g, ad_f2_g] # vector is a list, no shape, len(vector) = 3

orthogonal_vector = gram_schmidt2(vector)
# orthogonal_vector = sp.simplify(orthogonal_vector)
print(f"\nOrthogonal check:")
product = sp.simplify((orthogonal_vector.transpose() * ad_f_g)[0, 0].num)
print(product)


# TAG introduce rational polynomial p
monos = [x1, x2, x3, x1**2, x2**2, x3**2, x2*x3]
n_coeffs = len(monos)

theta = sp.IndexedBase('theta')

# Numerator uses x[0] ... x[n_coeffs-1]
poly_n = sum(theta[i] * monos[i] for i in range(n_coeffs))
# Denominator uses x[n_coeffs] ... x[2*n_coeffs - 1]
poly_d = sum(theta[i + n_coeffs] * monos[i] for i in range(n_coeffs))

poly_p = ur.UncanceledRational(poly_n, poly_d)

# Compute p*v
grad_vector = poly_p * orthogonal_vector # Bug fixed by adding RationalMatrix instance to class UncalceledRational 

J_grad = grad_vector.jacobian(variable_x)

# TODO implement for all diagonal entry
constraints = []
for i in range(J_grad.shape[0]):
    for j in range(i + 1, J_grad.shape[1]):

        diff = sp.expand(J_grad[i, j].num) - sp.expand(J_grad[j, i].num)

        # Convert to a polynomial in x1, x2 and x3
        poly = sp.Poly(diff, list(variable_x))

        # Get the coefficient dictionary
        coeff_dict = poly.as_dict()

        # # Collect terms with respect to powers of x2 and x3
        # collected = sp.collect(diff, [x1, x2, x3], evaluate=False)

        for _, coef in coeff_dict.items():
            if coef != 0:
                constraints.append(coef)

# TAG norm constraint on both num and den
c_norm1 = 0; c_norm2 = 0
for i in range(n_coeffs):
    c_norm1 += theta[i] ** 2
    c_norm2 += theta[i + n_coeffs] **2
c_norm1 -= 1; c_norm2 -= 1

constraints.append(c_norm1)
constraints.append(c_norm2)

constraint_funcs = [sp.lambdify(theta, expr, "numpy") for expr in constraints]

def total_violation(theta_vals):
    """
    Returns the sum of squared constraint violations
    """
    violations = constraint_violations2(
        constraint_funcs=constraint_funcs,
        theta_vals=theta_vals
    )
    return np.sum(violations**2)

# SUPTAG solver
# Number of random starting points
n_starts = 300
tolerance = 1e-6
solutions = []
best_f = float('inf') # np.float64
best_theta = None # np.ndarray, shape = (2*n_coeffs, )

known_solutions = [
        # Solution family 1: [0, 0, 0, 0, cos(α), sin(α)]
        np.array([0, 0, 0, 0, 0.9939, 0.1104, 0,
                  0, 0, 0, 0, 0, 0, -1]),  # α = 0
        np.array([0, 0, 0, 0, -0.9939, -0.1104, 0,
                  0, 0, 0, 0, 0, 0, 1]),  # α = π/2
    ]

for i in range(n_starts):
    num_coeff = random_unit_vector(n_coeffs)
    den_coeff = random_unit_vector(n_coeffs)
    initial_guess = np.concatenate([num_coeff, den_coeff])

    result = minimize(
        total_violation,
        initial_guess,
        method='SLSQP',
        options={'ftol': 1e-10, 'disp': False, 'maxiter': 1000}
    )

    if result.success and total_violation(result.x) < tolerance:
        violation_value = total_violation(result.x)
        
        # Track the best solution
        if violation_value < best_f:
            best_f = violation_value
            best_theta = result.x.copy()
        
        # Check if this solution is already found
        is_new = True
        for sol in solutions:
            if np.linalg.norm(result.x - sol) < 0.01:
                is_new = False
                break
        
        if is_new:
            print(f"\nFound new solution (iteration {i+1}):")
            print(result.x)
            verify_solution2(constraint_funcs, result.x)
            solutions.append(result.x)

print(f"\nTotal unique solutions found: {len(solutions)}")

cleaned_theta = np.where(np.abs(best_theta) < 1e-4, 0, best_theta)

# Output function
num_expr = sum(coef * mono for coef, mono in zip(cleaned_theta[:len(cleaned_theta)//2], monos))
den_expr = sum(coef * mono for coef, mono in zip(cleaned_theta[len(cleaned_theta)//2:], monos))
poly_p_expr = num_expr/den_expr

orth_vec_sp = orthogonal_vector.to_sympy_matrix()
orth_vec = [orth_vec_sp[i] for i in range(orth_vec_sp.shape[0])]
grad_vec = [poly_p_expr * expr for expr in orth_vec]
grad_vec_sp = sp.Matrix([poly_p_expr * expr for expr in orth_vec])
curl = is_curl_free(grad_vec_sp, (x1, x2, x3))

# TAG integration result
# h = symbolic_integration(grad_vec_sp, [x1, x2, x3]) 
h = symbolic_integration(sp.nsimplify(sp.expand(grad_vec_sp)), [x1, x2, x3]) # type = <class 'sympy.core.add.Add'>
with open('results.txt', 'w') as f:
    f.write(f"Total unique solutions found: {len(solutions)}\n\n")
    
    if best_theta is not None:
        f.write(f"Best solution (lowest violation):\n")
        f.write(f"best_theta: {best_theta}\n")
        f.write(f"best_f (violation): {best_f}\n\n")
    
    f.write("All solutions:\n")
    for i, sol in enumerate(solutions):
        f.write(f"Solution {i+1}:\n")
        f.write(f"{sol}\n")
        f.write(f"Violation: {total_violation(sol)}\n\n")

    f.write("Learned h function:\n")
    f.write(f"{h}\n")