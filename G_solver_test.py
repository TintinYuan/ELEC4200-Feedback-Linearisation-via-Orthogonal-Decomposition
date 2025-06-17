import sympy as sp
import numpy as np
from sympy.polys.monomials import itermonomials
from utils import gram_schmidt, jacobian, lie_bracket, func_chooser
from BH import BasinHopping_normed

x1, x2, x3 = sp.symbols('x1 x2 x3')
variable_x = sp.Matrix([x1, x2, x3])

fx, gx = func_chooser(2)

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
print(orthogonal_vector)

print(f"\nOrthogonal check:")
product = sp.simplify(orthogonal_vector.dot(ad_f_g))
print(product)

# SUPTAG Define the scalar polynomial p(theta, x)
# HACK For known known condition!!
# Generate all monomials of total degree <= n
monos = [x2**2, x3**2, x2*x3]
n_coeffs = len(monos) # Number of terms in monos

x = sp.IndexedBase('x') # Using indexed based variables instead of coeffs_n and coeffs_d

# Numerator uses x[0] ... x[n_coeffs-1]
poly_n = sum(x[i] * monos[i] for i in range(n_coeffs))
# Denominator uses x[n_coeffs] ... x[2*n_coeffs - 1]
poly_d = sum(x[i + n_coeffs] * monos[i] for i in range(n_coeffs))

poly_p = poly_n/poly_d

# Compute p*v
grad_vector = poly_p * orthogonal_vector
# grad_vector = sp.simplify(grad_vector)

J_grad = jacobian(grad_vector, variable_x)

# SUPTAG Define Frobenius norm-based loss function

# TAG Create data points
num_points = 10
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
    print(f"Processing point {i+1}/{num_points}", end='\r')

    # Substitute values into Jacobion
    J_subs = J_grad.subs(subs_dict)

    antisymmetric_part = J_subs - J_subs.T
    loss_term = antisymmetric_part.norm('fro')

    total_loss += loss_term

# TAG lambdified function
f_loss = sp.lambdify(x, total_loss)

# SUPTAG Optimise!
print("===Basin hopping on f_loss===")
initial_point = np.random.randn(2*len(monos))
initial_point /= np.linalg.norm(initial_point)  # ensure feasible initial guess


bh = BasinHopping_normed(
    objective_func=f_loss,
    initial_x=initial_point,
    temperature=10,
    step_size=1.5,
    max_iter=100
)

best_theta, best_f = bh.optimize()

with open('results.txt', 'w') as f:
    f.write(f"best_theta: {best_theta}\n")
    f.write(f"best_f: {best_f}\n")

print(f"Best funciton value: f = {best_f:.3f}")