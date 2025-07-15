import sympy as sp
from sympy import symbols

def zero_small_coefficients(expr, threshold=1e-5):
    """
    Replace coefficients with absolute value less than threshold with zero
    in a SymPy expression object
    """
    # If the expression is just a number
    if expr.is_Number:
        return sp.Integer(0) if abs(float(expr)) < threshold else expr
    
    # For Add expressions (sums of terms)
    if expr.is_Add:
        return sp.Add(*[zero_small_coefficients(term, threshold) for term in expr.args])
    
    # For Mul expressions (products of factors)
    elif expr.is_Mul:
        # Extract the coefficient and the rest of the expression
        coeff, rest = expr.as_coeff_Mul()
        if abs(float(coeff)) < threshold:
            return sp.Integer(0)
        return coeff * zero_small_coefficients(rest, threshold)
    
    # For expressions with powers, functions, etc.
    elif expr.args:
        new_args = [zero_small_coefficients(arg, threshold) for arg in expr.args]
        return expr.func(*new_args)
    
    # For atomic expressions like symbols
    return expr

# Example usage
x2, x3 = symbols('x2 x3')

# Let's assume this is your SymPy expression (not a string)
expr = -5.0901473181669e-7 * x2**2 * sp.log(9*x2**2 + x3**2) - 1.98761456308722 * x2**2 - 0.662538131138547 * x3**2

# Apply the function
simplified_expr = zero_small_coefficients(expr)

print("Original expression:")
print(expr)
print("\nSimplified expression (coefficients < 1e-5 set to zero):")
print(simplified_expr)