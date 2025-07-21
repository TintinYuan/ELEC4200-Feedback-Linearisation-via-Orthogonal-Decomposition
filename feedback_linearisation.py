import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def linearized_system(t, z):
    # At time t, get the corresponding x from the original system
    # We need this to compute the correct v
    z1, z2, z3 = z
    # Chain of integrators dynamics
    # Transform the input
    v_val = -z1 - 3*z2 - 3*z3
    dz1dt = z2
    dz2dt = z3
    dz3dt = v_val
    
    return np.array([dz1dt, dz2dt, dz3dt])

if __name__ == "__main__":
    # Simulation parameters
    t_span = (0, 15)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    x0 = np.array([1.0, 1.0, 1.0])

    sol_lin = solve_ivp(
        linearized_system,
        t_span,
        x0,
        t_eval=t_eval,
        method='RK45'
    )

    z_trajectory = np.zeros((3, len(sol_lin.t)))

    plt.figure(figsize=(10, 8))

    # Plot original system states
    plt.subplot(1, 1, 1)
    plt.plot(sol_lin.t, sol_lin.y[0], 'r', label='z₁')
    plt.plot(sol_lin.t, sol_lin.y[1], 'g', label='z₂')
    plt.plot(sol_lin.t, sol_lin.y[2], 'b', label='z₃')
    plt.xlabel('Time t')
    plt.ylabel('States')
    plt.title('Original System States with state feedback')
    plt.grid(True)
    plt.legend()
    plt.show()

