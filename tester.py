import sympy as sp

x1, x2, x3 = sp.symbols("x1 x2 x3")
x_variables = sp.Matrix([x1, x2, x3])

num = sp.UnevaluatedExpr((x1 + x2)) * sp.UnevaluatedExpr(x1)
den = sp.UnevaluatedExpr((x1 + x2))

f = num/den
f = sp.Matrix([f])
print(f)

Jf = f.jacobian(x_variables)
print(Jf)