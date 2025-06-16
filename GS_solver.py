import numpy as np
import sympy as sp
from sympy import Matrix, GramSchmidt, latex, pprint, diff, integrate

from utils import gram_schmidt


def dot_product_check(u, V):
    """
    Check the dot product of u dot [g, ad_fg, ... , ad_f^{n-2}g]

    input:
        u: Orthogonal vector computed via gram-schmidt
        V: Original n by n matrix [g, ad_fg, ... , ad_f^{n-1}g]

    output:
        all_zero: bool value of all zero or not
        vals: n-1 by 1 vector of dot product values
    """
    n_cols = V.shape[1] if hasattr(V, 'shape') else len(V)
    vals = []

    for i in range(n_cols - 1):
        val = u.dot(V[i])
        vals.append(sp.simplify(val))

    all_zero = all(val == 0 for val in vals)

    return all_zero, vals


'''
System1: 
f(x) = | x_2                        |   g(x) = | 0 |
       | -a*sin(x_1) - b(x_1 - x_3) |          | 0 |
       | x_4                        |          | 0 |
       | c(x_1 - x_3)               |          | d |
'''

a, b, c, d = sp.symbols('a b c d')
x_1, x_2, x_3, x_4 = sp.symbols('x1 x2 x3 x4')

# vectors = [Matrix([0, 0, 0, d]), Matrix([0, 0, -d, 0]), Matrix([0, b*d, 0, -c*d]), Matrix([-b*d, 0, c*d, 0])]


'''
System2:
f(x) = | -a*x_1                 |  g(x) = | 1 |
       | -b*x_2 + k - c*x_1*x_3 |         | 0 |
       | theta*x_1*x_2          |         | 0 |

'''

a, b, c, theta, k = sp.symbols('a b c theta k')
x_1, x_2, x_3 = sp.symbols('x1 x2 x3')

vectors = [Matrix([1, 0, 0]), Matrix([a, c*x_3, -theta*x_2]), Matrix([a**2, (a+b)*c*x_3, (b-a)*theta*x_2 - theta*k])]

orthogonal_vector1 = gram_schmidt(vectors)


# Convert to Matrix for easier handling
grad_vector = Matrix(orthogonal_vector1)

print("Check the dot product u dot [g, ad_fg, ... , ad_f^{n-2}g]")
all_zero, vals = dot_product_check(grad_vector, vectors)
print(f"\nIf all zero? {all_zero}")

# Print the gradient vector nicely
print("The gradient vector ∇h(x₁, x₂, x₃):")
pprint(grad_vector)
print("\nComponents of the gradient:")
print(f"∂h/∂x₁ = {grad_vector[0]}")
print(f"∂h/∂x₂ = {grad_vector[1]}")
print(f"∂h/∂x₃ = {grad_vector[2]}")

# Check if vector is conservative (curl-free)
# For a conservative vector field, mixed partial derivatives must be equal
print("\nChecking if the vector is conservative:")

# We need to replace the symbolic components with functions of x₁, x₂, x₃
# Let's assume the components already contain these variables
components = [grad_vector[i] for i in range(3)]

# Calculate mixed partial derivatives
curl_test = [
    sp.diff(components[2], x_2) - sp.diff(components[1], x_3),  # ∂/∂x₂(∂h/∂x₃) - ∂/∂x₃(∂h/∂x₂)
    sp.diff(components[0], x_3) - sp.diff(components[2], x_1),  # ∂/∂x₃(∂h/∂x₁) - ∂/∂x₁(∂h/∂x₃)
    sp.diff(components[1], x_1) - sp.diff(components[0], x_2)   # ∂/∂x₁(∂h/∂x₂) - ∂/∂x₂(∂h/∂x₁)
]

curl_test = [sp.simplify(curl) for curl in curl_test]

print("Curl components (should all be zero for a conservative field):")
for i, curl in enumerate(curl_test):
    print(f"Curl component {i+1}: {curl}")

is_conservative = all(curl == 0 for curl in curl_test)
print(f"\nIs the vector field conservative? {is_conservative}")

# If conservative, reconstruct the function by integration
if is_conservative:
    print("\nReconstructing the function h(x₁, x₂, x₃):")
    

    h = integrate(components[0], x_1)
    print("\nAfter integrating ∂h/∂x₁ with respect to x₁:")
    pprint(h)
    
    # Check if we need to add functions of x₂ and x₃
    # By differentiating h with respect to x₂, we should get components[1]
    correction_x2 = sp.simplify(components[1] - diff(h, x_2))
    if correction_x2 != 0:
        print(f"\nNeeded correction term (function of x₂ and x₃): {correction_x2}")
        # Integrate the correction term with respect to x₂
        correction_term = integrate(correction_x2, x_2)
        h = h + correction_term
        print("\nAfter adding correction term:")
        pprint(h)
    
    # Finally verify by differentiating
    verification = [
        sp.simplify(diff(h, x_1) - components[0]),
        sp.simplify(diff(h, x_2) - components[1]),
        sp.simplify(diff(h, x_3) - components[2])
    ]
    
    print("\nVerification (should all be zero):")
    for i, verify in enumerate(verification):
        print(f"∂h/∂x{i+1} - original component: {verify}")
    
    print("\nReconstructed function h(x₁, x₂, x₃):")
    pprint(h)
else:
    print("\nThe vector field is not conservative, so there is no function h whose gradient is the given vector.")

latex_repr = latex(Matrix(orthogonal_vector1))


simplified = [sp.simplify(orthogonal_vector1[i]) for i in range(len(orthogonal_vector1))]
print("Orthogonal vector (simplified):")
for i, comp in enumerate(simplified):
    pprint(comp)


# print("Orthogonal vector (calculated):\n", orthogonal_vector1)

