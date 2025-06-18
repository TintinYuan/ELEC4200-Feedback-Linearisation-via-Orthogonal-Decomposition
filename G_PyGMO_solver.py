import sympy as sp
import numpy as np
from sympy.polys.monomials import itermonomials
from utils import gram_schmidt, jacobian, lie_bracket, func_chooser
import pygmo as pg

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

# Generate all monomials of total degree <= n
mono_degree = 2
monos = itermonomials(variable_x, mono_degree)
monos = sorted(monos, key=lambda m: m.sort_key())
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
print()
print("===PyGMO on f_loss===")
initial_point = np.random.randn(2*n_coeffs)
initial_point /= np.linalg.norm(initial_point)  # ensure feasible initial guess

# TAG Define the PyGMO problem

class FrobeniusObj:
    """
    Function class for PyGMO solver
    """
    def __init__(self, obj_func, norm, dim):

        self.obj_func = obj_func
        self.norm = norm
        self.dim = dim

    def _eq_constraint_n(self, x):
        return np.sum(x[:len(x)//2]**2) - self.norm**2
    
    def _eq_constraint_d(self, x):
        return np.sum(x[len(x)//2:]**2) - self.norm**2

    def fitness(self, x):
        # Objection function
        obj = self.obj_func(x)

        # Equality constraints
        eq_cons = [
            self._eq_constraint_n(x),
            self._eq_constraint_d(x)
        ]

        # Combine objective and constraints
        return [obj] + eq_cons

    def get_bounds(self):
        """
        Defines the lower and upper bounds for each variable.
        Using -np.inf and np.inf for unbounded variables.
        """
        lower_bounds = [-self.norm] * self.dim
        upper_bounds = [self.norm] * self.dim
        return (lower_bounds, upper_bounds)

    def get_nec(self):
        return 2

    def get_nobj(self):
        return 1

prob_instance = pg.problem(
    FrobeniusObj(f_loss, 2, n_coeffs*2)
)

algo = pg.algorithm(pg.nlopt("cobyla"))
pop = pg.population(prob_instance, size=50) # Increased population size
pop = algo.evolve(pop)

champion_x = pop.champion_x
champion_f = pop.champion_f

# bh = BasinHopping_normed(
#     objective_func=f_loss,
#     initial_x=initial_point,
#     temperature=10,
#     norm=2.0,
#     step_size=1.0,
#     max_iter=200
# )

# best_theta, best_f = bh.optimize()

with open('results.txt', 'w') as f:
    f.write(f"best_theta: {champion_x}\n")
    f.write(f"best_f: {champion_f}\n")

print()
if isinstance(champion_f, (list, np.ndarray)):
    obj_value = champion_f[0]  # First element is the objective
    print(f"Best function value: f = {obj_value:.3f}")
    
    # Optionally, show constraint violations too
    if len(champion_f) > 1:
        constraints = champion_f[1:]
        print(f"Constraint violations: {constraints}")
else:
    print(f"Best function value: f = {champion_f:.3f}")