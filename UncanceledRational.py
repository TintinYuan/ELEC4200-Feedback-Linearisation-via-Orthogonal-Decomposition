import sympy as sp
from utils import gram_schmidt, lie_bracket, func_chooser

# Define symbolic variables
x1, x2, x3 = sp.symbols('x1 x2 x3')
variable_x = sp.Matrix([x1, x2, x3])

class UncanceledRational:
    """
    Class for handling rational functions without automatic cancellation of common factors.
    Allows for custom denominator forms during differentiation.
    """
    def __init__(self, num, den):
        """Initialize with numerator and denominator"""
        self.num = num
        self.den = den
    
    def __str__(self):
        """String representation of the rational function"""
        return f"({self.num})/({self.den})"
    
    def get_cancelled(self):
        """Return the simplified/cancelled form"""
        return sp.simplify(self.num / self.den)
        
    def diff(self, var, custom_denominator=None):
        """
        Differentiate with respect to a variable, with optional custom denominator
        """
        # Calculate derivative components
        dn_dx = sp.diff(self.num, var)
        dd_dx = sp.diff(self.den, var)
        
        # Standard quotient rule
        std_num = self.den * dn_dx - self.num * dd_dx
        std_den = self.den**2
        
        # Use standard result if no custom denominator
        if custom_denominator is None:
            return UncanceledRational(std_num, std_den)
            
        # Adjust numerator to maintain mathematical correctness with custom denominator
        adjustment_factor = custom_denominator / std_den
        adjusted_num = sp.expand(std_num * adjustment_factor)
        
        return UncanceledRational(adjusted_num, custom_denominator)
        
    def jacobian(self, variables, custom_denominator=None):
        """
        Compute Jacobian matrix of partial derivatives
        """
        jac = [self.diff(var, custom_denominator) for var in variables]
        
        # Return as a list, not as a Matrix since SymPy can't handle UncanceledRational objects in Matrix
        return jac
        
    def subs(self, *args, **kwargs):
        """Substitute values into the expression"""
        return UncanceledRational(
            self.num.subs(*args, **kwargs),
            self.den.subs(*args, **kwargs)
        )
        
if __name__ == "__main__":
    # Create example reducible rational functions
    den = x1**2 + x2**2
    num = (x2 + x2*x3) * den
    rf_example = UncanceledRational(num, den)

    print("--- Demonstration of UncanceledRational Class ---")
    print(f"Original reducible function: {rf_example}")
    print(f"Cancelled form: {rf_example.get_cancelled()}")

    # Example with differentiation and custom denominator
    print("\n--- Differentiation with Custom Denominator ---")

    # Create a rational function with x2 + x2*x3 in numerator
    # This should give us the form you requested after differentiation
    rf = UncanceledRational(num, den)
    print(f"Original function: {rf}")

    # Set up the target denominator (x1^2 + x2^2)^2
    target_denominator = (x1**2 + x2**2)**2  # Expanded: x1^4 + 2*x1^2*x2^2 + x2^4

    # Compute derivative with custom denominator
    df_dx1 = rf.diff(x1, custom_denominator=target_denominator)
    print(f"\ndf/dx1 with custom denominator: {df_dx1}")
    print(f"Expanded numerator: {sp.expand(df_dx1.num)}")
    print(f"Expanded denominator: {sp.expand(df_dx1.den)}")

    # Compute Jacobian with the same custom denominator
    print("\n--- Jacobian with Custom Denominator ---")
    jacobian = rf.jacobian([x1, x2, x3], custom_denominator=target_denominator)
    for i, j in enumerate(jacobian):
        print(f"∂f/∂x{i+1} = {j}")

    # Show how to substitute values
    print("\n--- Substituting Values ---")
    subs_result = df_dx1.subs({x1: 1, x2: 2, x3: 3})
    print(f"df/dx1 at (1,2,3): {subs_result}")
    print(f"Numerator: {subs_result.num}")
    print(f"Denominator: {subs_result.den}")

    # Example with more complex function
    print("\n--- Advanced Example ---")
    f1 = UncanceledRational(x1*x3, x1**2 + x2**2)
    f2 = UncanceledRational(x2*x3, (x1**2 + x2**2)**2)

    print(f"Function 1: {f1}")
    print(f"Function 2: {f2}")
    print(f"df1/dx1 with standard denominator: {f1.diff(x1)}")
    print(f"df1/dx1 with custom denominator (x1^2+x2^2)^2: {f1.diff(x1, (x1**2 + x2**2)**2)}")