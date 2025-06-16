import sympy as sp
import numpy as np
from sympy.polys.monomials import itermonomials
from utils import gram_schmidt, jacobian, lie_bracket, func_chooser
from BH import BasinHopping

x1, x2, x3 = sp.symbols('x1 x2 x3')
variable_x = sp.Matrix([x1, x2, x3])

fx, gx = func_chooser(1)

# SUPTAG Deriving the orthogonal vector
# Step 1: Compute ad_f g = [f, g]
ad_f_g = lie_bracket(fx, gx, variable_x)
ad_f_g = sp.simplify(ad_f_g)
print("ad_f g = [f, g] =")
print(ad_f_g)
print()

# Step 2: Compute ad_f² g = ad_f(ad_f g) = [f, [f, g]]
ad_f2_g = lie_bracket(fx, ad_f_g, variable_x)
ad_f2_g = sp.sympify(ad_f2_g)
print("ad_f² g = [f, [f, g]] =")
print(ad_f2_g)
print()

vector = [gx, ad_f_g, ad_f2_g]
# print(vector)

orthogonal_vector = gram_schmidt(vector)
orthogonal_vector = sp.simplify(orthogonal_vector)

print(f"\nOrthogonal vector:")
# print(orthogonal_vector)

print(f"\nOrthogonal check:")
product = sp.simplify(orthogonal_vector.dot(ad_f_g))
print(product)

# SUPTAG Define the scalar polynomial p(theta, x)

# Generate all monomials of total degree <= n
mono_degree = 3
monos = itermonomials(variable_x, mono_degree)
monos = sorted(monos, key=lambda m: m.sort_key())

# Generate coefficient of the same length (a0 as theta)
coeffs_n = sp.symbols(f'n0:{len(monos)}') # For numerator
coeffs_d = sp.symbols(f'd0:{len(monos)}') # For denominator

# Generate Polynomial from monomials
poly_n = sum(c * m for c, m in zip(coeffs_n, monos))
poly_d = sum(c * m for c, m in zip(coeffs_d, monos))

poly_p = poly_n/poly_d

# Compute p*v
grad_vector = poly_p * orthogonal_vector
# grad_vector = sp.simplify(grad_vector)

J_grad = jacobian(grad_vector, variable_x)

# SUPTAG Define Frobenius norm-based loss function

# TAG Create data points
num_points = 1
total_loss = 0

x_min = -2; x_max = 2

data_points = []
for i in range(num_points):
    # Generate random values for demonstration
    point = {
        x1: np.random.uniform(x_min, x_max),
        x2: np.random.uniform(x_min, x_max),
        x3: np.random.uniform(x_min, x_max)
    }
    data_points.append(point)

for i, subs_dict in enumerate(data_points):
    print(f"Processing point {i+1}/{num_points}")

    # Substitute values into Jacobion
    J_subs = J_grad.subs(subs_dict)

    antisymmetric_part = J_subs - J_subs.T
    loss_term = antisymmetric_part.norm('fro')

    total_loss += loss_term

# TAG lambdified function
x = sp.IndexedBase('x')
replacements = {
    coeff: x[i] for i, coeff in enumerate(coeffs_n + coeffs_d)
}
total_loss_indexed = total_loss.subs(replacements)
f_loss = sp.lambdify(x, total_loss_indexed, modules='numpy')

# SUPTAG Optimise!
print("===Basin hopping on f_loss===")
initial_point = np.ones((1, len(coeffs_n) + len(coeffs_d))).flatten()

bh = BasinHopping(
    objective_func=f_loss,
    initial_x=initial_point,
    temperature=10,
    step_size=1,
    max_iter=1000
)

best_theta, best_f = bh.optimize()

print(f"Best funciton value: f = {best_f:.3f}")