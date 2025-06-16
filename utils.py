import sympy as sp

def gram_schmidt(V):
    """
    Compute the vector that is orthogonal to the first n-1 columns of V

    input:
        V: n by n matrix consist of [g, ad_fg, ad_f^2g, ... , ad_f^{n-1}g]

    output: 
        dh: n by 1 vector that is orthogonal to V[:-1]
    """
    n_cols = V.shape[1] if hasattr(V, 'shape') else len(V)
    U = []

    for i in range(n_cols):
        u = V[i]
        for j in range(i):
            uj = U[j]
            vi = V[i]
            dot_uj_vi = sum(uj[j] * vi[j] for j in range(len(vi)))
            dot_uj_uj = sum(uj[j] * uj[j] for j in range(len(uj)))
            proj = (dot_uj_vi/dot_uj_uj)*uj
            u = u - proj
        U.append(u)
    return U[-1]


def jacobian(vector_field, variables):
    """Compute the Jacobian matrix of a vector field."""
    return vector_field.jacobian(variables)

def lie_bracket(f, g, variables):
    """Compute the Lie bracket [f, g] = (∂g/∂x)f - (∂f/∂x)g"""
    Jf = jacobian(f, variables)
    Jg = jacobian(g, variables)
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
            
            gx = sp.Matrix([1,
                            0,
                            0])
            
            return fx, gx

        case _:
            fx = sp.Matrix([-x1,
                            -2*x2 - x1*x3,
                            3*x1*x2])
            
            gx = sp.Matrix([1,
                            0,
                            0])
            
            return fx, gx