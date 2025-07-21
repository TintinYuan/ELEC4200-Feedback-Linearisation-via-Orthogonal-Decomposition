import sympy as sp
import numpy as np
import UncanceledRational as ur
# from UncanceledRational import UncanceledRational as ur
# from UncanceledRational import RationalMatrix as urMatrix

def gram_schmidt(G):
    """
    Compute the vector that is orthogonal to the first n-1 columns of V

    input:
        V: n by n matrix consist of [g, ad_fg, ad_f^2g, ... , ad_f^{n-1}g]

    output: 
        dh: n by 1 vector that is orthogonal to V[:-1]
    """
    n_cols = G.shape[1] if hasattr(G, 'shape') else len(G)
    V = []

    for i in range(n_cols):
        v = G[i]
        for j in range(i):
            vj = V[j]
            gi = G[i]
            dot_vj_gi = sum(vj[j] * gi[j] for j in range(len(gi)))
            dot_vj_vj = sum(vj[j] * vj[j] for j in range(len(vj)))
            proj = (dot_vj_gi/dot_vj_vj)*vj
            v = v - proj
        V.append(v)
    return V[-1]

# Ur version of gram_schmidt
def gram_schmidt2(G):

    n_cols = G.shape[1] if hasattr(G, 'shape') else len(G)
    V = [] # constructed orthogonal matrix

    for i in range(n_cols):
        v = G[i] # current column of G = [g, adfg, adf2g, ...]
        for j in range(i):
            vj = V[j]
            gi = G[i] # RationalMatrix vector

            dot_vj_gi = (vj.transpose() * gi)[0, 0]
            
            dot_vj_vj = (vj.transpose() * vj)[0, 0]

            proj = vj * (dot_vj_gi/dot_vj_vj)

            v = v - proj

        V.append(v)



    return V[-1]

def jacobian(vector_field, variables):
    """Compute the Jacobian matrix of a vector field."""
    return vector_field.jacobian(variables)

def lie_bracket(f, g, variables):
    """Compute the Lie bracket [f, g] = (∂g/∂x)f - (∂f/∂x)g"""
    Jf = jacobian(f, variables)
    Jg = jacobian(g, variables)
    return Jg * f - Jf * g

# Ur version of lie_bracket
def lie_bracket2(f, g, variables):

    Jf = f.jacobian(variables)
    Jg = g.jacobian(variables)
    return Jg * f - Jf * g
# TAG Function chooser

def func_chooser(num):
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    # a = 1; b = 2; c = 1; theta = 3; k = 0
    a = 1; b = 2; c = 1; theta = 1; k = 2
    match num:
        case 1:
            fx = sp.Matrix([-x1 + x1*x2,
                            -2*x2 - 2*x1*x3,
                            3*x2*x3])

            gx = sp.Matrix([0,
                            x1,
                            1])
            
            return fx, gx
            
        case 2:
            fx = sp.Matrix([-a * x1,
                            -b*x2 + k - c * x1*x3,
                            theta*x1*x2])
            
            gx = sp.Matrix([1,# x1*x2
                            0, # x2
                            0])# 0
            
            return fx, gx

        case _:
            fx = sp.Matrix([-x1,
                            -2*x2 - x1*x3,
                            3*x1*x2])
            
            gx = sp.Matrix([1,
                            0,
                            0])
            
            return fx, gx
        
# Ur version of func_chooser
def func_chooser2(num):
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    # a = 1; b = 2; c = 1; theta = 3; k = 0
    a = 1; b = 2; c = 1; theta = 3; k = 2
    match num:
        case 1:
            fx = ur.RationalMatrix([
                [ur.UncanceledRational(-x1 + x1*x2)],
                [ur.UncanceledRational(-2*x2 - 2*x1*x3)],
                [ur.UncanceledRational(3*x1*x2)]
            ])

            gx = ur.RationalMatrix([
                [ur.UncanceledRational(0)],
                [ur.UncanceledRational(x1)],
                [ur.UncanceledRational(1)]
            ])
            return fx, gx
        case 2:
            fx = ur.RationalMatrix([
                [ur.UncanceledRational(-a * x1)],
                [ur.UncanceledRational(-b*x2 + k - c * x1*x3)],
                [ur.UncanceledRational(theta*x1*x2)]
            ])

            gx = ur.RationalMatrix([
                [ur.UncanceledRational(1)],
                [ur.UncanceledRational(0)],
                [ur.UncanceledRational(0)]
            ])
            return fx, gx
        case _: # Same as case 2
            fx = ur.RationalMatrix([
                [ur.UncanceledRational(-x1)],
                [ur.UncanceledRational(-2*x2 - x1*x3)],
                [ur.UncanceledRational(3*x1*x2)]
            ])

            gx = ur.RationalMatrix([
                [ur.UncanceledRational(1)],
                [ur.UncanceledRational(0)],
                [ur.UncanceledRational(0)]
            ])
            return fx, gx

def constraint_violations2(constraint_funcs, theta_vals):
    
    results = [func(theta_vals) for func in constraint_funcs]

    return np.array(results)

def verify_solution2(constraint_funcs, theta_vals):
    """
    Verify that a solution satisfies all constraints
    """
    violations = constraint_violations2(constraint_funcs, theta_vals)
    max_violation = np.max(np.abs(violations))
    
    print(f"Max constraint violation: {max_violation:.6e}")
    
    if max_violation < 1e-3:
        print("All constraints are satisfied!")
    else:
        print("Constraints are not satisfied.")
        for i, v in enumerate(violations):
            print(f"Constraint {i}: {v:.6e}")

def clean_small_coeffs(expr, tolerance=1e-4):
    """
    Replace coefficients smaller than tolerance with zero in a SymPy expression
    
    Args:
        expr: A SymPy expression
        tolerance: Threshold below which coefficients are set to zero
        
    Returns:
        A SymPy expression with small coefficients removed
    """
    if expr.is_Add:
        return sp.Add(*[clean_small_coeffs(term, tolerance) for term in expr.args])
    elif expr.is_Mul:
        coeff, rest = expr.as_coeff_mul()
        if abs(float(coeff)) < tolerance:
            return sp.S.Zero
        else:
            return coeff * sp.Mul(*rest)
    elif hasattr(expr, "is_number") and expr.is_number:
        if abs(float(expr)) < tolerance:
            return sp.S.Zero
        return expr
    else:
        return expr

def zero_small_coefficients(expr, threshold=1e-5):
    """
    Replace coefficients with absolute value less than threshold with zero
    in a SymPy expression object.
    
    Args:
        expr (sympy.Expr or numeric): The expression to process
        threshold (float): Threshold below which coefficients are zeroed
        
    Returns:
        sympy.Expr: Expression with small coefficients removed
    """
    # Handle non-expression inputs
    if expr is None:
        return None
    
    # Handle Python numeric types (int, float, complex)
    if isinstance(expr, (int, float, complex)):
        if abs(expr) < threshold:
            return sp.Integer(0)
        return sp.sympify(expr)
    
    # If the expression is just a number
    if expr.is_Number:
        try:
            return sp.Integer(0) if abs(float(expr)) < threshold else expr
        except (ValueError, TypeError):
            # Handle special values like Infinity or symbolic numbers
            return expr
    
    # For Add expressions (sums of terms)
    if expr.is_Add:
        result = sp.Integer(0)
        for term in expr.args:
            processed_term = zero_small_coefficients(term, threshold)
            if processed_term != 0:
                result += processed_term
        return result
    
    # For Mul expressions (products of factors)
    if expr.is_Mul:
        # Process each factor individually rather than recursively calling on 'rest'
        factors = expr.as_ordered_factors()
        processed_factors = []
        
        for factor in factors:
            processed_factor = zero_small_coefficients(factor, threshold)
            if processed_factor == 0:
                # If any factor is zero, the entire product is zero
                return sp.Integer(0)
            processed_factors.append(processed_factor)
        
        # Multiply all processed factors together
        if processed_factors:
            return sp.Mul(*processed_factors)
        return sp.Integer(1)  # Empty product is 1
    
    # For Pow expressions (powers)
    if expr.is_Pow:
        base = zero_small_coefficients(expr.base, threshold)
        if base == 0:
            # 0^anything is 0 (except 0^0, which is 1)
            if expr.exp == 0:
                return sp.Integer(1)
            return sp.Integer(0)
        # Process the exponent only if it's not a simple integer
        if not expr.exp.is_Integer:
            exp = zero_small_coefficients(expr.exp, threshold)
            return base ** exp
        return base ** expr.exp
    
    # For expressions with other structures (functions, etc.)
    if expr.args:
        new_args = [zero_small_coefficients(arg, threshold) for arg in expr.args]
        return expr.func(*new_args)
    
    # For symbols and other atomic expressions
    return expr


# Symbolic integration
def symbolic_integration(grad_vec, vars):
    """
    Integration of the symbolic vector field

    Args:
        grad_vec (list): A list of Jacobian gradient vector
        vars (sp.IndexedBase): State variables
    Returns:
        (sp.expr): Output funciton h
    """

    dim = len(grad_vec)

    h = 0
    for i in range(dim):
        h = zero_small_coefficients(h)
        grad = grad_vec[i]
        remaining = grad - sp.diff(h, vars[i])
        remaining = zero_small_coefficients(remaining)
        # remaining = sp.simplify(sp.expand(remaining))
        # remaining = clean_small_coeffs(remaining, tolerance=1e-4)
        h += sp.integrate(remaining, vars[i])

    h = h.evalf()
    simplified_h = zero_small_coefficients(h, threshold=1e-4)
    # Result testing 
    # for i in range(dim):
    #     assert sp.expand(sp.diff(h, vars[i]) - grad_vec[i]) == 0, \
    #         f"Verification failed for component {i}"

    print("Original scalar function h(x) =", simplified_h)
    return simplified_h

def is_curl_free(F, vars=None):
    """
    Check if a 3D vector field F is curl-free.
    F: sympy.Matrix of shape (3, 1) or (3,)
    vars: tuple/list of sympy symbols (x1, x2, x3), default to (x1, x2, x3)
    Returns: True if curl is zero vector, else False
    """
    if vars is None:
        x1, x2, x3 = sp.symbols('x1 x2 x3')
        vars = (x1, x2, x3)
    if isinstance(F, list):
        F = sp.Matrix(F)
    curl = sp.Matrix([
        sp.diff(F[2], vars[1]) - sp.diff(F[1], vars[2]),
        sp.diff(F[0], vars[2]) - sp.diff(F[2], vars[0]),
        sp.diff(F[1], vars[0]) - sp.diff(F[0], vars[1])
    ])
    # Check if all components are zero
    return [sp.simplify(comp) for comp in curl]


def linear_dynamics(dim, h, variables):

    Ac = sp.Matrix(dim, dim, lambda i, j: 1 if j == i+1 else 0) # sp.Matrix, shape = (dim, dim)

    Bc = sp.Matrix(dim, 1, lambda i, j: 1 if i == dim-1 else 0) # sp.Matrix, shape = (dim, 1)

    
    
    return