import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable, Tuple, List

class BasinHopping:
    def __init__(self, 
                 objective_func: Callable,
                 initial_x: np.ndarray,
                 temperature: float = 1.0,
                 step_size: float = 0.5,
                 max_iter: int = 100,
                 local_method: str = 'L-BFGS-B'):
        """
        Basin Hopping Algorithm Implementation
        
        Parameters:
        - objective_func: Function to minimize
        - initial_x: Starting point
        - temperature: Controls acceptance probability
        - step_size: Size of random perturbations
        - max_iter: Maximum number of iterations
        - local_method: Local optimization method
        """
        self.objective_func = objective_func
        self.x = initial_x.copy()
        self.temperature = temperature
        self.step_size = step_size
        self.max_iter = max_iter
        self.local_method = local_method
        
        # Track optimization history
        self.history = []
        self.best_x = initial_x.copy()
        self.best_f = float('inf')
        
    def _local_minimize(self, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform local minimization from given starting point"""
        result = minimize(self.objective_func, x0, method=self.local_method)
        return result.x, result.fun
    
    def _accept_reject(self, f_new: float, f_current: float) -> bool:
        """Monte Carlo acceptance criterion"""
        if f_new < f_current:
            return True
        else:
            prob = np.exp(-(f_new - f_current) / self.temperature)
            return np.random.random() < prob
    
    def _perturb(self, x: np.ndarray) -> np.ndarray:
        """Apply random perturbation to current position"""
        return x + self.step_size * np.random.randn(len(x))
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run the basin hopping algorithm"""
        # Initial local minimization
        self.x, f_current = self._local_minimize(self.x)
        self.best_x = self.x.copy()
        self.best_f = f_current
        
        self.history.append((self.x.copy(), f_current, True))
        
        for i in range(self.max_iter):
            # Perturb current position
            x_trial = self._perturb(self.x)
            
            # Local minimization from perturbed position
            x_min, f_min = self._local_minimize(x_trial)
            
            # Accept or reject the new minimum
            accepted = self._accept_reject(f_min, f_current)
            
            if accepted:
                self.x = x_min
                f_current = f_min
                
                # Update global best if improved
                if f_min < self.best_f:
                    self.best_x = x_min.copy()
                    self.best_f = f_min
            
            self.history.append((x_min.copy(), f_min, accepted))
            
            # Optional: adaptive temperature cooling
            # self.temperature *= 0.99

            # Optimising pricess
            interval = self.max_iter//100
            if((i+1)%interval == 0):
                print(f"Optimising process {(i+1)/self.max_iter * 100:.1f}%", end='\r')
        
        return self.best_x, self.best_f

# Test Functions
def rastrigin(x):
    """Rastrigin function - highly multimodal test function"""
    A = 10
    n = len(x)
    return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def ackley(x):
    """Ackley function - another multimodal test function"""
    a, b, c = 20, 0.2, 2*np.pi
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e

def himmelblau(x):
    """Himmelblau's function - has 4 global minima"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Example Usage
if __name__ == "__main__":
    # Test on Himmelblau's function (2D)
    print("=== Basin Hopping on Himmelblau's Function ===")
    initial_point = np.array([0.0, 0.0])
    
    bh = BasinHopping(
        objective_func=himmelblau,
        initial_x=initial_point,
        temperature=10.0,
        step_size=2.0,
        max_iter=50
    )
    
    best_x, best_f = bh.optimize()
    
    print(f"Best solution found: x = [{', '.join(f'{x:.3f}' for x in best_x)}]")
    print(f"Best function value: f = {best_f:.3f}")
    
    # Known global minima of Himmelblau's function
    known_minima = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126)
    ]
    
    print("\nKnown global minima:")
    for i, (x1, x2) in enumerate(known_minima):
        print(f"  Minimum {i+1}: ({x1:.3f}, {x2:.3f}) -> f = {himmelblau(np.array([x1, x2])):.3f}")
    
    # Test on Rastrigin function (higher dimensional)
    print("\n=== Basin Hopping on 5D Rastrigin Function ===")
    dim = 5
    initial_point = np.random.randn(dim) * 2
    
    bh_rast = BasinHopping(
        objective_func=rastrigin,
        initial_x=initial_point,
        temperature=3.0,
        step_size=0.2,
        max_iter=1000
    )
    
    best_x_rast, best_f_rast = bh_rast.optimize()
    
    print(f"Initial point: x_init = [{', '.join(f'{x:.3f}' for x in initial_point)}]")
    print(f"Best solution found: x = [{', '.join(f'{x:.3f}' for x in best_x_rast)}]")
    print(f"Best function value: f = {best_f_rast:.3f}")
    print(f"Global minimum is at origin with f = 0")
    
    # TAG Visualization for 2D case (rastrigin)
    print("\n=== Basin Hopping on 2D Rastrigin Function ===")
    dim = 2
    initial_point = np.random.randn(dim) * 2
    bh_rast2 = BasinHopping(
        objective_func=rastrigin,
        initial_x=initial_point,
        temperature=5,
        step_size=0.2,
        max_iter=200
    )

    best_x_rast2, best_f_rast2 = bh_rast2.optimize()

    plt.figure(figsize=(12, 5))
    
    # Plot 1: Function landscape with optimization path
    plt.subplot(1, 2, 1)
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = rastrigin([X1, X2])
    
    plt.contour(X1, X2, Z, levels=50, alpha=0.6)
    plt.colorbar(label='Function Value')
    
    # Plot accepted points
    accepted_points = [(x, f) for x, f, accepted in bh_rast2.history if accepted]
    if accepted_points:
        accepted_x = np.array([x for x, f in accepted_points])
        plt.plot(accepted_x[:, 0], accepted_x[:, 1], 'ro-', alpha=0.7, 
                label='Accepted Points', markersize=4)
    
    # Plot best point
    plt.plot(best_x_rast2[0], best_x_rast2[1], '*', markersize=15, color="hotpink", label='Best Solution')

    print(f"Initial point: x_init = [{', '.join(f'{x:.3f}' for x in initial_point)}]")
    print(f"Best solution found: x = [{', '.join(f'{x:.3f}' for x in best_x_rast2)}]")
    print(f"Best function value: f = {best_f_rast2:.3f}")
    print(f"Global minimum is at origin with f = 0")
    
    # Plot known minima
    known_minima_rast2 = [(0.0, 0.0)]
    for i, (x1_min, x2_min) in enumerate(known_minima_rast2):
        plt.plot(x1_min, x2_min, 'bs', markersize=8, alpha=0.7)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Basin Hopping on Rastrigin Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Convergence history
    plt.subplot(1, 2, 2)
    f_values = [f for x, f, accepted in bh_rast2.history]
    accepted_mask = [accepted for x, f, accepted in bh_rast2.history]
    
    plt.plot(f_values, 'b-', alpha=0.5, label='All Evaluations')
    accepted_indices = [i for i, accepted in enumerate(accepted_mask) if accepted]
    accepted_f_values = [f_values[i] for i in accepted_indices]
    plt.plot(accepted_indices, accepted_f_values, 'ro-', label='Accepted Points')
    
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Convergence History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTotal function evaluations: {len(bh_rast2.history)}")
    print(f"Acceptance rate: {sum(accepted for _, _, accepted in bh_rast2.history) / len(bh_rast2.history):.2%}")