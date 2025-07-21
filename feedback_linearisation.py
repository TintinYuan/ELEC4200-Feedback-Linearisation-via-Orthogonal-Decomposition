from utils import symbolic_integration
import sympy as sp

if __name__ == "__main__":
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    grad_vec = sp.Matrix([x1**2, x1*x2, x1*x3])
    h = symbolic_integration(grad_vec, [x1, x2, x3])
    print(h)