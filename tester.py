import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dynamics(t, x, u_func):
    """
    Control affine system dynamics: dx/dt = f(x) + g(x)*u
    
    Args:
        t: Time (scalar)
        x: State vector [x1, x2, x3]
        u_func: Control input function u(t)
    
    Returns:
        State derivative vector
    """
    x1, x2, x3 = x
    
    # f(x) component
    fx = np.array([
        -x1,
        -2*x2 - x1*x3,
        3*x1*x2
    ])
    
    # g(x) component
    gx = np.array([
        1,
        0,
        0
    ])
    
    # Control input at time t
    u = u_func(t)
    
    # Control affine system: dx/dt = f(x) + g(x)*u
    return fx + gx * u

def sinusoidal_input(t, amplitude=1.0, frequency=1.0):
    """
    Sinusoidal control input u(t) = A*sin(ω*t)
    """
    return amplitude * np.sin(frequency * t)

# Simulation parameters
t_span = (0, 10)  # Time span for simulation
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points to evaluate
x0 = [1.0, 0.5, 0.2]  # Initial state

# Input parameters
input_amplitude = 1.0
input_frequency = 2.0  # radians/second

# Create control input function with specified parameters
u_func = lambda t: sinusoidal_input(t, input_amplitude, input_frequency)

# Solve the system of ODEs
sol = solve_ivp(
    lambda t, x: dynamics(t, x, u_func),
    t_span,
    x0,
    t_eval=t_eval,
    method='RK45'  # 4th-order Runge-Kutta method
)

# Plot the results
plt.figure(figsize=(15, 10))

# Plot state trajectories
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], 'r', label='x₁')
plt.plot(sol.t, sol.y[1], 'g', label='x₂')
plt.plot(sol.t, sol.y[2], 'b', label='x₃')
plt.xlabel('Time t')
plt.ylabel('States')
plt.title('State Trajectories')
plt.grid(True)
plt.legend()

# Plot control input
plt.subplot(2, 1, 2)
u_values = [u_func(t) for t in sol.t]
plt.plot(sol.t, u_values, 'k')
plt.xlabel('Time t')
plt.ylabel('Control Input u(t)')
plt.title(f'Sinusoidal Control Input: u(t) = {input_amplitude} sin({input_frequency}t)')
plt.grid(True)

plt.tight_layout()
plt.savefig('control_affine_system_simulation.png')
plt.show()

# Print final states
print(f"Final states at t = {sol.t[-1]}:")
print(f"x₁ = {sol.y[0, -1]}")
print(f"x₂ = {sol.y[1, -1]}")
print(f"x₃ = {sol.y[2, -1]}")