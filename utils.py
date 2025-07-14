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
            fx = sp.Matrix([-x1,
                            -2*x2 - x1*x3,
                            3*x1*x2])
            
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
        grad = grad_vec[i]
        remaining = grad - sp.diff(h, vars[i])
        h += sp.integrate(remaining, vars[i])

    # Result testing 
    for i in range(dim):
        assert sp.expand(sp.diff(h, vars[i]) - grad_vec[i]) == 0, \
            f"Verification failed for component {i}"


    print("Original scalar function h(x) =", h)
    return h