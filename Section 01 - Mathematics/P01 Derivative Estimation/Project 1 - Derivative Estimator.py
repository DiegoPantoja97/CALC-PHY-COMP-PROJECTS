#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Define the mathematical function here
def f(x):
    A = 1.0       # Amplitude
    mu = 0.0      # Mean
    sigma = 1.0   # Standard deviation
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Function to define the tangent line
def tangent_line(x, x0, slope, f_value):
    """
    Tangent line function.
    :param x: The variable
    :param x0: The point of tangency
    :param slope: The slope of the tangent line
    :param f_value: The function value at x0
    :return: The tangent line value at x
    """
    return slope * (x - x0) + f_value

# Function for estimating derivatives
def derivative_estimator():
    # Input point and step size
    x0 = float(input("Enter the point x0: "))
    h = float(input("Enter the step size h: "))

    # Derivative approximations
    forward_diff = (f(x0 + h) - f(x0)) / h
    backward_diff = (f(x0) - f(x0 - h)) / h
    symmetric_diff = (f(x0 + h) - f(x0 - h)) / (2 * h)
    high_order_diff = (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)

    # Print results
    print(f"Forward Difference: {forward_diff}")
    print(f"Backward Difference: {backward_diff}")
    print(f"Symmetric Difference: {symmetric_diff}")
    print(f"High Order Method: {high_order_diff}")

    # Use the best approximation for the tangent line slope
    tangent_slope = symmetric_diff
    f_value_at_x0 = f(x0)

    # Plot the function and tangent line
    wide_range_x = np.linspace(x0 - 5, x0 + 5, 500)
    short_range_x = np.linspace(x0 - 2 * 100*h, x0 + 2 * 100*h, 1000)    
    y = f(wide_range_x)

    plt.figure(figsize=(8, 6))
    plt.plot(wide_range_x, y, label='Function f(x)')
    plt.plot(short_range_x, tangent_line(short_range_x, x0, tangent_slope, f_value_at_x0),label=f'Tangent at x0={x0}')
    plt.scatter([x0], [f_value_at_x0], color='red', label=f'Point x0={x0}')
    plt.title("Function and Tangent Line")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


while True:
    derivative_estimator()
    repeat = input("Would you like to compute the derivative at another point? (yes/no): ").strip().lower()
    if repeat != 'yes':
        break


# In[17]:


""" Segment - 2 Higher Order Derivative Estimator """


import numpy as np
import matplotlib.pyplot as plt


# Define the mathematical function here
def f(x):
    A = 1.0       # Amplitude
    mu = 0.0      # Mean
    sigma = 1.0   # Standard deviation
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# First Derivative
def first_derivative(f, x0, h):
    """
    Compute the first derivative using the central difference method.
    :param f: Function to differentiate.
    :param x0: Point at which the derivative is estimated.
    :param h: Step size.
    :return: Estimated first derivative.
    """
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

# Second Derivative
def second_derivative(f, x0, h):
    """
    Compute the second derivative using the central difference method.
    :param f: Function to differentiate.
    :param x0: Point at which the derivative is estimated.
    :param h: Step size.
    :return: Estimated second derivative.
    """
    return (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / (h ** 2)

# Third Derivative
def third_derivative(f, x0, h):
    """
    Compute the third derivative using the central difference method.
    :param f: Function to differentiate.
    :param x0: Point at which the derivative is estimated.
    :param h: Step size.
    :return: Estimated third derivative.
    """
    return (f(x0 + 2*h) - 2*f(x0 + h) + 2*f(x0 - h) - f(x0 - 2*h)) / (2 * h ** 3)

# Fourth Derivative
def fourth_derivative(f, x0, h):
    """
    Compute the fourth derivative using the central difference method.
    :param f: Function to differentiate.
    :param x0: Point at which the derivative is estimated.
    :param h: Step size.
    :return: Estimated fourth derivative.
    """
    return (f(x0 + 2*h) - 4*f(x0 + h) + 6*f(x0) - 4*f(x0 - h) + f(x0 - 2*h)) / (h ** 4)

# Main Program
if __name__ == "__main__":
    # Input parameters
    x0 = float(input("Enter the point x0: "))
    h = float(input("Enter the step size h: "))
    
    # Compute selected derivatives
    print("\nEstimated Derivatives:")
    
    # First Derivative
    first = first_derivative(f, x0, h)
    print(f"1st derivative: {first:.6f}")
    
    # Second Derivative
    second = second_derivative(f, x0, h)
    print(f"2nd derivative: {second:.6f}")
    
    # Third Derivative
    third = third_derivative(f, x0, h)
    print(f"3rd derivative: {third:.6f}")
    
    # Fourth Derivative
    fourth = fourth_derivative(f, x0, h)
    print(f"4th derivative: {fourth:.6f}")



# In[ ]:





# In[ ]:




