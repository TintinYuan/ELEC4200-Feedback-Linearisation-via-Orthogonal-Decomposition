import sympy as sp

# Define symbolic variables
x1, x2, x3, a, b, c, k, theta = sp.symbols('x1 x2 x3 a b c k theta')

# Define the common denominator D
D = c**2 * x3**2 + theta**2 * x2**2

# Define the first polynomial expression (now without division by theta*x2)
P = c*x3*(a + b) - c*x3*(c**2*x3**2*(a + b) - theta*x2*(-k*theta + theta*x2*(-a + b)))/D

# Define the second polynomial expression (now without division by c*x3)
Q = -k*theta + theta*x2*(-a + b) + theta*x2*(c**2*x3**2*(a + b) - theta*x2*(-k*theta + theta*x2*(-a + b)))/D

# Simplify each expression and convert to a common denominator
P_simplified = sp.simplify(P)
Q_simplified = sp.simplify(Q)

print("First polynomial simplified:")
print(P_simplified)
print("\nSecond polynomial simplified:")
print(Q_simplified)

# Convert to standard form (numerator/denominator)
P_standard = sp.together(P_simplified)
Q_standard = sp.together(Q_simplified)

print("\nFirst polynomial in standard form:")
print(P_standard)
print("\nSecond polynomial in standard form:")
print(Q_standard)

# Extract numerator and denominator
P_num, P_den = sp.fraction(P_standard)
Q_num, Q_den = sp.fraction(Q_standard)

print("\nP numerator:", P_num)
print("P denominator:", P_den)
print("\nQ numerator:", Q_num)
print("Q denominator:", Q_den)

# Calculate GCD of numerators
num_gcd = sp.gcd(P_num, Q_num)
print("\nGCD of numerators:", num_gcd)

# Calculate LCM of denominators
den_lcm = sp.lcm(P_den, Q_den)
print("LCM of denominators:", den_lcm)

# The GCD is the GCD of numerators divided by LCM of denominators
gcd_result = num_gcd / den_lcm
print("\nGCD of the polynomials:")
print(sp.simplify(gcd_result))

simplified_P = sp.simplify(P/gcd_result)
simplified_Q = sp.simplify(Q/gcd_result)

print("\nSimplified results:")
print(f"Simplified P: {simplified_P}")
print(f"Simplified Q: {simplified_Q}")