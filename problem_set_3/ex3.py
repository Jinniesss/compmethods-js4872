import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**3
x0 = 3
true_derivative = 3 * x0**2
h_values = np.logspace(-10, 0, num=1000)
numerical_approx = (f(x0 + h_values) - f(x0)) / h_values
difference = np.abs(numerical_approx - true_derivative)

plt.figure(figsize=(10, 6))
plt.loglog(h_values, difference)
plt.xlabel("h (step size)")
plt.ylabel("Absolute Value of the Difference")
plt.title("Derivative Approximation Error")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.savefig("problem_set_3/derivative_error.png")