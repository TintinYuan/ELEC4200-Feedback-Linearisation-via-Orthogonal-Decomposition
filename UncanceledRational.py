import sympy as sp

# TAG class UncancelRational
class UncanceledRational:
    """
    Class for handling rational functions without automatic cancellation of common factors.
    Allows for custom denominator forms during differentiation.
    """
    def __init__(self, num, den=1):
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
    
    def subs(self, *args, **kwargs):
        """Substitute values into the expression"""
        return UncanceledRational(
            self.num.subs(*args, **kwargs),
            self.den.subs(*args, **kwargs)
        )
    
    # TODO add LCM
    def __add__(self, other):
        """
        Add two UncanceledRational objects or an UncanceledRational and a scalar.
        
        For fractions a/b + c/d, we use LCM of denominators:
        lcm = least common multiple of b and d
        a/b + c/d = (a*lcm/b + c*lcm/d)/lcm
        """
        if isinstance(other, UncanceledRational):
            # When adding two rational functions, use LCM of denominators
            try:
                # Try to compute LCM (works for polynomial expressions in SymPy)
                lcm_den = sp.lcm(self.den, other.den)
                
                # Calculate factors for each numerator
                self_factor = sp.simplify(lcm_den / self.den)
                other_factor = sp.simplify(lcm_den / other.den)
                
                # Calculate new numerator with the LCM denominator
                new_num = self.num * self_factor + other.num * other_factor
                new_den = lcm_den
            except Exception:
                # Fall back to product of denominators if LCM fails
                new_num = self.num * other.den + self.den * other.num
                new_den = self.den * other.den
                
            return UncanceledRational(new_num, new_den)
        else:
            # When adding a scalar (converting it to rational with denominator 1)
            new_num = self.num + self.den * other
            new_den = self.den
            return UncanceledRational(new_num, new_den)
    
    def __radd__(self, other):
        """Handle addition when the UncanceledRational is on the right side"""
        # For example, when doing 5 + rational_obj
        if other == 0:
            # Special case for sum() function which starts with 0
            return self
        else:
            # Delegate to __add__
            return self.__add__(other)
            
    def __sub__(self, other):
        """
        Subtract another UncanceledRational or scalar from this one.
        
        For fractions a/b - c/d, we use LCM of denominators:
        lcm = least common multiple of b and d
        a/b - c/d = (a*lcm/b - c*lcm/d)/lcm
        """
        if isinstance(other, UncanceledRational):
            # When subtracting two rational functions, use LCM of denominators
            try:
                # Try to compute LCM (works for polynomial expressions in SymPy)
                lcm_den = sp.lcm(self.den, other.den)
                
                # Calculate factors for each numerator
                self_factor = lcm_den / self.den
                other_factor = lcm_den / other.den
                
                # Calculate new numerator with the LCM denominator
                new_num = self.num * self_factor - other.num * other_factor
                new_den = lcm_den
            except Exception:
                # Fall back to product of denominators if LCM fails
                new_num = self.num * other.den - self.den * other.num
                new_den = self.den * other.den
                
            return UncanceledRational(new_num, new_den)
        else:
            # When subtracting a scalar
            new_num = self.num - self.den * other
            new_den = self.den
            return UncanceledRational(new_num, new_den)
    
    def __rsub__(self, other):
        """Handle subtraction when the UncanceledRational is on the right side"""
        # For example, when doing 5 - rational_obj
        new_num = other * self.den - self.num
        new_den = self.den
        return UncanceledRational(new_num, new_den)
    
    def __mul__(self, other):
        """
        Multiply this UncanceledRational with another or with a scalar.
        
        For fractions a/b * c/d = (ac)/(bd)
        """
        if isinstance(other, UncanceledRational):
            # When multiplying two rational functions
            new_num = self.num * other.num
            new_den = self.den * other.den
            return UncanceledRational(new_num, new_den)
        elif isinstance(other, RationalMatrix):
            return other * self
        else:
            # When multiplying by a scalar
            new_num = self.num * other
            new_den = self.den
            return UncanceledRational(new_num, new_den)
    
    def __rmul__(self, other):
        """Handle multiplication when the UncanceledRational is on the right side"""
        # This is commutative, so we can just call __mul__
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Divide this UncanceledRational by another or by a scalar.
        
        For fractions (a/b) / (c/d) = (ad)/(bc)
        """
        if isinstance(other, UncanceledRational):
            # When dividing two rational functions
            new_num = self.num * other.den
            new_den = self.den * other.num
            return UncanceledRational(new_num, new_den)
        else:
            # When dividing by a scalar
            new_num = self.num
            new_den = self.den * other
            return UncanceledRational(new_num, new_den)
    
    def __rtruediv__(self, other):
        """Handle division when the UncanceledRational is on the right side"""
        # For example, when doing 5 / rational_obj
        new_num = other * self.den
        new_den = self.num
        return UncanceledRational(new_num, new_den)
    
    def __neg__(self):
        """Negate the rational function"""
        return UncanceledRational(-self.num, self.den)
    
    def __pow__(self, power):
        """Raise the rational function to a power"""
        if not isinstance(power, int) and not isinstance(power, float):
            raise ValueError("Power must be a number")
            
        if power == 0:
            return UncanceledRational(1, 1)
        elif power > 0:
            return UncanceledRational(self.num**power, self.den**power)
        else:
            # Negative power means reciprocal
            return UncanceledRational(self.den**abs(power), self.num**abs(power))
    
    @classmethod
    def from_sympy(cls, expr):
        """
        Convert a SymPy expression to an UncanceledRational object.
        Properly extracts numerator and denominator for rational expressions.
        """
        try:
            # Try to extract numerator and denominator from SymPy expression
            num, den = expr.as_numer_denom()
            return cls(num, den)
        except (AttributeError, TypeError):
            # If not a fraction, use the expression as numerator with denominator 1
            return cls(expr)
        
# TAG class RaitionalMatrix
class RationalMatrix:
    """
    A matrix class for working with UncanceledRational objects.
    Supports basic matrix operations like addition, multiplication, and transpose.
    """
    def __init__(self, data=None, rows=0, cols=0):
        """
        Initialize a matrix of UncanceledRational objects.
        
        Args:
            data: List of lists or 2D array of values (can be UncanceledRational objects, numbers, or SymPy expressions)
            rows: Number of rows (only used if data is None)
            cols: Number of columns (only used if data is None)
        """
        if data is not None:
            self.rows = len(data)
            self.cols = len(data[0]) if self.rows > 0 else 0
            
            # Convert all elements to UncanceledRational if they're not already
            self.data = []
            for row in data:
                new_row = []
                for elem in row:
                    if isinstance(elem, UncanceledRational):
                        new_row.append(elem)
                    else:
                        # Use from_sympy method to properly handle SymPy expressions
                        new_row.append(UncanceledRational.from_sympy(elem))
                self.data.append(new_row)
        else:
            self.rows = rows
            self.cols = cols
            self.data = [[UncanceledRational(0) for _ in range(cols)] for _ in range(rows)]
    
    def __str__(self):
        """String representation of the matrix"""
        result = "[\n"
        for row in self.data:
            result += "  [" + ", ".join(str(elem) for elem in row) + "]\n"
        result += "]"
        return result
    
    def __getitem__(self, indices):
        """Access matrix elements using the [i, j] syntax"""
        if isinstance(indices, tuple) and len(indices) == 2:
            i, j = indices
            return self.data[i][j]
        else:
            return self.data[indices]
    
    @property
    def shape(self):
        """Return the shape of the matrix as a tuple (rows, cols)"""
        return (self.rows, self.cols)
    
    def __setitem__(self, indices, value):
        """Set matrix elements using the [i, j] syntax"""
        if not isinstance(value, UncanceledRational):
            value = UncanceledRational.from_sympy(value)
            
        if isinstance(indices, tuple) and len(indices) == 2:
            i, j = indices
            self.data[i][j] = value
        else:
            self.data[indices] = value
    
    def __add__(self, other):
        """Add two matrices"""
        if not isinstance(other, RationalMatrix):
            raise ValueError("Can only add RationalMatrix objects")
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")
        
        result = RationalMatrix(rows=self.rows, cols=self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] + other[i, j]
        
        return result
    
    def __sub__(self, other):
        """Subtract one matrix from another"""
        if not isinstance(other, RationalMatrix):
            raise ValueError("Can only subtract RationalMatrix objects")
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for subtraction")
        
        result = RationalMatrix(rows=self.rows, cols=self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] - other[i, j]
        
        return result
    
    def __neg__(self):
        """Negate the matrix (unary minus operator)"""
        result = RationalMatrix(rows=self.rows, cols=self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = -self[i, j]
        
        return result
    
    def __mul__(self, other):
        """
        Matrix multiplication or scalar multiplication
        
        A * B for matrices or A * s for scalar s
        """
        if isinstance(other, RationalMatrix):
            # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions incompatible for multiplication")
            
            result = RationalMatrix(rows=self.rows, cols=other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    sum_val = UncanceledRational(0)
                    for k in range(self.cols):
                        sum_val = sum_val + (self[i, k] * other[k, j])
                    result[i, j] = sum_val
            
            return result
        else:
            # Scalar multiplication
            result = RationalMatrix(rows=self.rows, cols=self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = self[i, j] * other
            
            return result
    
    def __rmul__(self, other):
        """Right scalar multiplication"""
        # This is just scalar multiplication, which is commutative
        return self.__mul__(other)
    
    def transpose(self):
        """Return the transpose of the matrix"""
        result = RationalMatrix(rows=self.cols, cols=self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        
        return result
    
    @classmethod
    def identity(cls, size):
        """Create an identity matrix of given size"""
        mat = cls(rows=size, cols=size)
        for i in range(size):
            mat[i, i] = UncanceledRational(1)
        return mat
    
    @classmethod
    def from_sympy_matrix(cls, sp_matrix):
        """Create a RationalMatrix from a SymPy Matrix"""
        rows = sp_matrix.rows
        cols = sp_matrix.cols
        
        data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                elem = sp_matrix[i, j]
                # Use the from_sympy method to properly extract numerator and denominator
                rational_elem = UncanceledRational.from_sympy(elem)
                row_data.append(rational_elem)
            data.append(row_data)
            
        return cls(data=data)
    
    def to_sympy_matrix(self):
        """Convert to a SymPy Matrix with cancelled fractions"""
        data = [[self[i, j].get_cancelled() for j in range(self.cols)] for i in range(self.rows)]
        return sp.Matrix(data)
    
    def jacobian(self, variables, custom_denominator=None):
        """
        Calculate the Jacobian matrix of a vector (n×1 matrix) of rational functions
        with respect to the given variables.
        
        Args:
            variables: List or array of variables to differentiate with respect to
            custom_denominator: Optional custom denominator to use for all elements
                               of the Jacobian matrix
        
        Returns:
            A new RationalMatrix representing the Jacobian matrix where
            J[i,j] = ∂self[i,0]/∂variables[j]
        
        Example:
            If self is a column vector [f₁; f₂; f₃] and variables = [x₁, x₂, x₃],
            the result will be:
            [∂f₁/∂x₁  ∂f₁/∂x₂  ∂f₁/∂x₃]
            [∂f₂/∂x₁  ∂f₂/∂x₂  ∂f₂/∂x₃]
            [∂f₃/∂x₁  ∂f₃/∂x₂  ∂f₃/∂x₃]
        """
        if self.cols != 1:
            raise ValueError("Jacobian can only be computed for a column vector (n×1 matrix)")
        
        # Create a matrix for the Jacobian with dimensions n×m
        # where n is the number of functions and m is the number of variables
        n = self.rows
        m = len(variables)
        jacobian_matrix = RationalMatrix(rows=n, cols=m)
        
        # Calculate each element of the Jacobian matrix
        for i in range(n):
            for j in range(m):
                # Get the function (an UncanceledRational object)
                func = self[i, 0]
                # Differentiate with respect to the variable
                derivative = func.diff(variables[j], custom_denominator)
                # Store in the Jacobian matrix
                jacobian_matrix[i, j] = derivative
        
        return jacobian_matrix
        
if __name__ == "__main__":
    # Define symbolic variables
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    variable_x = sp.Matrix([x1, x2, x3])
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
    
    # Demonstrate arithmetic operations
    print("\n--- Arithmetic Operations ---")
    r1 = UncanceledRational(x1, x1**2 + 1)
    r2 = UncanceledRational(x2, x2**2 + 1)
    
    print(f"r1 = {r1}")
    print(f"r2 = {r2}")
    
    # Addition
    r_sum = r1 + r2
    print(f"\nr1 + r2 = {r_sum}")
    print(f"Numerator: {sp.expand(r_sum.num)}")
    print(f"Denominator: {sp.expand(r_sum.den)}")
    
    # LCM example with polynomial denominators
    print("\n--- LCM Example ---")
    r3 = UncanceledRational(x1, (x1+1)*(x2+1))
    r4 = UncanceledRational(x2, (x1+1)*(x3+1))
    r_sum_lcm = r3 + r4
    print(f"r3 = {r3}")
    print(f"r4 = {r4}")
    print(f"r3 + r4 = {r_sum_lcm}")
    print(f"Numerator: {sp.expand(r_sum_lcm.num)}")
    print(f"Denominator: {sp.factor(r_sum_lcm.den)}") # Show factored form
    
    # Subtraction
    r_diff = r1 - r2
    print(f"\nr1 - r2 = {r_diff}")
    
    # Multiplication
    r_prod = r1 * r2
    print(f"\nr1 * r2 = {r_prod}")
    
    # Division
    r_div = r1 / r2
    print(f"\nr1 / r2 = {r_div}")
    
    # Scalar operations
    print(f"\n2 + r1 = {2 + r1}")
    print(f"r1 * 3 = {r1 * 3}")
    print(f"5 - r2 = {5 - r2}")
    print(f"10 / r1 = {10 / r1}")
    
    # Combining operations
    complex_expr = r1 * r2 + 2 * r1 / (r2 - 1)
    print(f"\nComplex expression r1*r2 + 2*r1/(r2-1) = {complex_expr}")
    
    # Matrix operations with UncanceledRational objects
    print("\n--- Matrix Operations with UncanceledRational ---")
    
    # Create matrices with UncanceledRational elements
    print("Creating matrices:")
    A = RationalMatrix([
        [UncanceledRational(x1, x1**2 + 1), UncanceledRational(x2)],
        [UncanceledRational(x3, 2), UncanceledRational(1, x1**2 + x2**2)]
    ])
    print(f"A = {A}")
    
    B = RationalMatrix([
        [UncanceledRational(2), UncanceledRational(x1*x2)],
        [UncanceledRational(x3, x1), UncanceledRational(3, 2)]
    ])
    print(f"B = {B}")
    
    # Matrix addition
    print("\nMatrix Addition:")
    C = A + B
    print(f"A + B = {C}")
    
    # Matrix subtraction
    print("\nMatrix Subtraction:")
    F = A - B
    print(f"A - B = {F}")
    
    # Matrix negation
    print("\nMatrix Negation:")
    G = -A
    print(f"-A = {G}")
    
    # Combination of operations
    print("\nCombined operation:")
    H = A - (-B)  # Should be equivalent to A + B
    print(f"A - (-B) = {H}")
    print(f"Is A - (-B) == A + B? {str(all(H[i, j].num == C[i, j].num and H[i, j].den == C[i, j].den for i in range(H.rows) for j in range(H.cols)))}")
    
    # Matrix multiplication
    print("\nMatrix Multiplication:")
    D = A * B
    print(f"A * B = {D}")
    print(f"First element expanded: ({sp.expand(D[0,0].num)})/({sp.expand(D[0,0].den)})")
    
    # Scalar multiplication
    print("\nScalar Multiplication:")
    E = 2 * A
    print(f"2 * A = {E}")
    
    # Matrix transpose
    print("\nMatrix Transpose:")
    AT = A.transpose()
    print(f"A^T = {AT}")
    
    # Create a matrix from a SymPy matrix
    print("\nConversion from/to SymPy Matrix:")
    sym_mat = sp.Matrix([
        [x1/x2, x3],
        [1, x1*x2/x3]
    ])
    print(f"SymPy Matrix: {sym_mat}")
    
    rat_mat = RationalMatrix.from_sympy_matrix(sym_mat)
    print(f"Converted to RationalMatrix: {rat_mat}")
    
    # Convert back to SymPy matrix (with cancellation)
    sym_mat2 = rat_mat.to_sympy_matrix()
    print(f"Converted back to SymPy Matrix: {sym_mat2}")
    
    # Test the from_sympy method specifically
    print("\n--- Testing from_sympy Method ---")
    expr1 = x1/x2
    expr2 = (x1**2 + x2**2)/(x3 + 1)
    expr3 = x1 * x2 * x3
    
    rational1 = UncanceledRational.from_sympy(expr1)
    rational2 = UncanceledRational.from_sympy(expr2)
    rational3 = UncanceledRational.from_sympy(expr3)
    
    print(f"Original expression: {expr1}")
    print(f"UncanceledRational: {rational1}")
    print(f"Numerator: {rational1.num}")
    print(f"Denominator: {rational1.den}")
    
    print(f"\nOriginal expression: {expr2}")
    print(f"UncanceledRational: {rational2}")
    print(f"Numerator: {rational2.num}")
    print(f"Denominator: {rational2.den}")
    
    print(f"\nOriginal expression: {expr3}")
    print(f"UncanceledRational: {rational3}")
    print(f"Numerator: {rational3.num}")
    print(f"Denominator: {rational3.den}")
    
    # Demonstrate the RationalMatrix.jacobian method
    print("\n--- RationalMatrix Jacobian Example ---")
    # Create a column vector of rational functions
    f1 = UncanceledRational(x1**2 * x2, x3 + 1)
    f2 = UncanceledRational(x1 * x3, x2**2)
    f3 = UncanceledRational(x2 * x3, x1 + x2)
    
    vector_f = RationalMatrix([
        [f1],
        [f2],
        [f3]
    ])
    
    print("Vector of rational functions f(x):")
    print(vector_f)
    
    # Calculate the Jacobian matrix with respect to [x1, x2, x3]
    vars_list = [x1, x2, x3]
    J = vector_f.jacobian(vars_list)
    
    print("\nJacobian matrix J = ∂f/∂x:")
    print(J)
    
    # Show individual elements
    print("\nIndividual elements of the Jacobian matrix:")
    for i in range(J.rows):
        for j in range(J.cols):
            var_name = vars_list[j]
            print(f"∂f{i+1}/∂{var_name} = {J[i, j]}")
    
    # Calculate Jacobian with a custom denominator
    print("\nJacobian with custom denominator (x1^2 + x2^2)^2:")
    custom_den = (x1**2 + x2**2)**2
    J_custom = vector_f.jacobian(vars_list, custom_denominator=custom_den)
    print(J_custom)
