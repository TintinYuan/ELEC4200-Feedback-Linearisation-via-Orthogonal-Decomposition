import sympy as sp
from utils import symbolic_integration

# For a 4D example
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
vars = [x1, x2, x3, x4]

# Example gradient field (should be a conservative field)
grad_vec = [
    2*x1 + x2,          # ∂h/∂x1
    x1 + 2*x2 + x3,     # ∂h/∂x2
    x2 + 2*x3 + x4,     # ∂h/∂x3
    x3 + 2*x4           # ∂h/∂x4
]

h = symbolic_integration(grad_vec, vars)
print()