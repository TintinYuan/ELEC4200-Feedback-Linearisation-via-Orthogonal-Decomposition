import sympy as sp
from sympy.polys.monomials import itermonomials
from utils import gram_schmidt, lie_bracket, func_chooser

x1, x2, x3 = sp.symbols('x1 x2 x3')
variable_x = sp.Matrix([x1, x2, x3])

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


