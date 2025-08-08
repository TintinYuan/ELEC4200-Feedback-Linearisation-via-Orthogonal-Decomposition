# Orthogonal Decomposition-based Feedback Linearisation

This repository provides tools for computing coordinate transforms for feedback linearisation using orthogonal decomposition methods. It features multiple solvers and utilities for reconstructing scalar functions from conservative vector fields and for symbolic polynomial operations.

## File Structure

- `PINNs/`: Physics-Informed Neural Networks for reconstructing scalar functions from conservative vector fields.

- `BH.py`: Basin Hopping demo script.

- `GS_solver.py`: Gram-Schmidt solver for computing orthogonal bases from $N$ linearly independent $N$-dimensional vector fields.

- `G_BH_solver.py`: Gram-Schmidt orthogonalisation with Basin Hopping solver for conservative field computation.
  
  - `G_BH_solver_test.py`: Simplified version for testing and demonstration.

- `G_CMA_solver.py`: Gram-Schmidt orthogonalisation with CMA (Covariance Matrix Adaptation) solver to compute conservative fields.
  
  - `G_CMA_solver_test.py`: Simplified version for testing and demonstration.

- `UncanceledRational.py`: Class for performing arithmetic operations ($+,-,\times,\div$ etc.) with reducible rational polynomials.

- <mark>`G_Optimisation_solver.py`</mark>: Gram-Schmidt orthogonalisation with bilinear optimisation solver for conservative fields (depends on `UncanceledRational.py`).

- `utils.py`: Utility functions for mathematical operations used in optimisation and orthogonalisation.