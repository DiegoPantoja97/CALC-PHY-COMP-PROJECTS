#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt

# Define the mathematical function here
def f(x):
    return np.exp(x)  # Example function, replace as needed

# Function to calculate the tangent line at a point
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

# Function to approximate the curve using tangent lines
def tangent_line_approximation(x_range, points, h):
    """
    Approximate the curve using tangent lines.
    :param x_range: Tuple containing (xmin, xmax)
    :param points: Number of tangent lines to use
    :param h: Step size for derivative calculation
    """
    x_min, x_max = x_range
    x_points = np.linspace(x_min, x_max, points)
    all_tangent_x = []
    all_tangent_y = []

    plt.figure(figsize=(10, 6))
    for x0 in x_points:
        # Calculate the slope using symmetric difference
        slope = (f(x0 + h) - f(x0 - h)) / (2 * h)
        f_value_at_x0 = f(x0)

        # Define tangent line range
        tangent_x = np.linspace(x0 - 0.1, x0 + 0.1, 10)
        tangent_y = tangent_line(tangent_x, x0, slope, f_value_at_x0)

        # Store tangent line points
        all_tangent_x.extend(tangent_x)
        all_tangent_y.extend(tangent_y)

        # Plot the tangent line
        plt.plot(tangent_x, tangent_y, '--')

    # Plot the original function for reference
    x = np.linspace(x_min, x_max, 1000)
    plt.plot(x, f(x), label="Function f(x)", color="black", linewidth=2)
    plt.title("Curve Approximation via Tangent Lines (Superposed)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot only the tangent line approximation
    plt.figure(figsize=(10, 6))
    plt.plot(all_tangent_x, all_tangent_y, label="Tangent Line Approximation", color="blue")
    plt.title("Tangent Line Approximation (Only)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

# Function to approximate the curve using scatter plot and linear interpolation
def scatter_plot_approximation(x_range, points):
    """
    Approximate the curve using scatter plot and linear interpolation.
    :param x_range: Tuple containing (xmin, xmax)
    :param points: Number of scatter points to use
    """
    x_min, x_max = x_range
    x_points = np.linspace(x_min, x_max, points)
    y_points = f(x_points)

    # Plot scatter points and interpolation (superposed)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_points, y_points, color="red", label="Scatter Points")
    plt.plot(x_points, y_points, '-', label="Linear Interpolation", color="blue")
    x = np.linspace(x_min, x_max, 1000)
    plt.plot(x, f(x), label="Function f(x)", color="black", linewidth=2)
    plt.title("Curve Approximation via Scatter Plot and Interpolation (Superposed)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot only the linear interpolation
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, '-', label="Linear Interpolation (Only)", color="blue")
    plt.scatter(x_points, y_points, color="red", label="Scatter Points")
    plt.title("Linear Interpolation (Only)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

# Main Program to Run Both Methods
if __name__ == "__main__":
    # Input range and number of points
    x_range = (-5, 10)  # Range of x-values
    points = 50  # Number of tangent lines or scatter points
    h = 0.1  # Step size for derivative calculation

    # Tangent Line Approximation
    tangent_line_approximation(x_range, points, h)

    # Scatter Plot and Linear Interpolation
    scatter_plot_approximation(x_range, points)


# In[ ]:




