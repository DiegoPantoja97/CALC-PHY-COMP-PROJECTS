#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sympy as sp
import numpy as np

def propagate_errors():
    print("Welcome to the Error Propagation Tool!")
    
    # Define the function here
    # Example: f(x, y) = x + y
    x, y = sp.symbols('x y')  # Define symbols for variables
    f = x + y   # Define the function here

    print(f"The defined function is: {f}")

    # Input arrays for measured values and uncertainties
    num_vars = int(input("Enter the number of variables (e.g., 3 for x, y, z): "))

    # Validate that the number of variables matches the function
    variables = [x, y][:num_vars]  # Adjust variable list based on input count

    values = list(map(float, input(f"Enter the measured values as a space-separated list (e.g., 2.0 3.0 4.0 for {variables}): ").split()))
    uncertainties = list(map(float, input(f"Enter the uncertainties as a space-separated list (e.g., 0.1 0.2 0.15 for {variables}): ").split()))

    if len(values) != num_vars or len(uncertainties) != num_vars:
        print("Error: The number of values and uncertainties must match the number of variables!")
        return

    # Calculate partial derivatives
    partial_derivatives = [sp.diff(f, var) for var in variables]

    # Calculate propagated uncertainty
    propagated_uncertainty = 0
    for i, var in enumerate(variables):
        partial_value = partial_derivatives[i].subs(dict(zip(variables, values)))
        propagated_uncertainty += (partial_value * uncertainties[i])**2

    propagated_uncertainty = np.sqrt(float(propagated_uncertainty))

    # Evaluate the function at the given values
    function_value = float(f.subs(dict(zip(variables, values))))

    print("\nResults:")
    print(f"The evaluated function value is: {function_value}")
    print(f"The propagated uncertainty is: {propagated_uncertainty}")

# Run the program
propagate_errors()


# In[ ]:




