# G_BH solver with uncancelled denominator

import sympy as sp
import numpy as np
from utils import gram_schmidt2, lie_bracket2, func_chooser2
import UncanceledRational as ur

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
monos = [x2**2, x3**2, x2*x3]
n_coeffs = len(monos)

theta = sp.IndexedBase('theta')

# Numerator uses x[0] ... x[n_coeffs-1]
poly_n = sum(theta[i] * monos[i] for i in range(n_coeffs))
# Denominator uses x[n_coeffs] ... x[2*n_coeffs - 1]
poly_d = sum(theta[i + n_coeffs] * monos[i] for i in range(n_coeffs))

poly_p = ur.UncanceledRational(poly_n, poly_d)

# Compute p*v
grad_vector = orthogonal_vector * poly_p

J_grad = grad_vector.jacobian(variable_x)

diff = sp.expand(J_grad[2, 1].num) - sp.expand(J_grad[1, 2].num)

# Collect terms with respect to powers of x2 and x3
collected = sp.collect(diff, [x1, x2, x3], evaluate=False)

# Extract constraints (coefficients must be zero for equality)
constraints = []
for term, coef in collected.items():
    if coef != 0:
        constraint = sp.Eq(coef, 0)
        constraints.append((term, constraint))

# Print constraints
for term, constraint in constraints:
    print(f"For term {term}:")
    print(f"  {constraint}")

# Try to solve the system
solution = sp.solve(constraints, theta)
print("\nSolution for theta values:")
print(solution)