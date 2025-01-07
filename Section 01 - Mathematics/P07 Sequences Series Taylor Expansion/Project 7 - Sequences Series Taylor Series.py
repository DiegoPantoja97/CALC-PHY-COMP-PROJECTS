#!/usr/bin/env python
# coding: utf-8

# In[25]:


""" Segment 1 Sequence and Series Investigation """

import numpy as np
import matplotlib.pyplot as plt

# Recursive Sequence Generator
def generate_recursive_sequence(n_terms, initial_terms, recurrence_relation):
    """
    Generate a sequence recursively.
    
    :param n_terms: Number of terms in the sequence.
    :param initial_terms: List of initial terms.
    :param recurrence_relation: A function defining the recurrence relation.
    :return: List of terms in the sequence.
    """
    sequence = list(initial_terms)
    for n in range(len(initial_terms), n_terms):
        sequence.append(recurrence_relation(sequence, n))
    return sequence

# Explicit Sequence Generator
def generate_explicit_sequence(n_terms, explicit_formula):
    """
    Generate a sequence explicitly as a function of n.
    
    :param n_terms: Number of terms in the sequence.
    :param explicit_formula: A function defining the explicit formula.
    :return: List of terms in the sequence.
    """
    return [explicit_formula(n) for n in range(n_terms)]

# Finite Summation
def finite_sum(sequence):
    """
    Compute the finite summation of a sequence.
    
    :param sequence: List of terms in the sequence.
    :return: Sum of the sequence.
    """
    return sum(sequence)

# Recursive Relation for Coefficients
def generate_coefficients(n_terms, initial_terms, recurrence_relation):
    """
    Generate coefficients recursively based on a recurrence relation.

    :param n_terms: Number of terms to generate.
    :param initial_terms: List of initial coefficients.
    :param recurrence_relation: Lambda function defining the recurrence relation.
    :return: List of coefficients.
    """
    coefficients = list(initial_terms)
    for n in range(len(initial_terms), n_terms):
        coefficients.append(recurrence_relation(coefficients, n))
    return coefficients

# Table of Index vs Coefficient Term
def print_coefficient_table(coefficients):
    """
    Print a table of index vs coefficient term.

    :param coefficients: List of coefficients.
    """
    print(f"{'Index (n)':>10} | {'Coefficient':>15}")
    print("-" * 30)
    for n, coef in enumerate(coefficients):
        print(f"{n:>10} | {coef:>15.6f}")

# Scatter Plot for Sequence
def scatter_plot_sequence(coefficients):
    """
    Plot the sequence as a scatter plot.

    :param coefficients: List of coefficients.
    """
    indices = range(len(coefficients))
    plt.figure(figsize=(8, 6))
    plt.scatter(indices, coefficients, color="blue", label="Coefficient Values")
    plt.title("Scatter Plot of Sequence")
    plt.xlabel("Index (n)")
    plt.ylabel("Coefficient")
    plt.grid()
    plt.legend()
    plt.show()

# Scatter Plot for Series
def scatter_plot_series(coefficients):
    """
    Plot the partial sums of the series as a scatter plot.

    :param coefficients: List of coefficients.
    """
    partial_sums = np.cumsum(coefficients)
    indices = range(len(partial_sums))
    plt.figure(figsize=(8, 6))
    plt.scatter(indices, partial_sums, color="red", label="Partial Sums")
    plt.title("Scatter Plot of Series")
    plt.xlabel("Index (n)")
    plt.ylabel("Partial Sum")
    plt.grid()
    plt.legend()
    plt.show()


# Main Program
if __name__ == "__main__":
    # Example 1: Fibonacci Sequence (Recursive)
    # Fibonacci Sequence and Golden Ratio
    print("\nFibonacci Sequence and Golden Ratio:")
    fibonacci_sequence = [0, 1]
    for _ in range(18):  # Generate more terms
        fibonacci_sequence.append(fibonacci_sequence[-1] + fibonacci_sequence[-2])
    print(f"Fibonacci Sequence: {fibonacci_sequence}")

    golden_ratios = [fibonacci_sequence[i] / fibonacci_sequence[i - 1] for i in range(2, len(fibonacci_sequence))]
    print(f"Golden Ratios (Last 5 Approximations): {golden_ratios[-5:]}")
    print(f"Converged to: {golden_ratios[-1]:.8f}")

    # Example 2: Arithmetic Sequence
    print("\nArithmetic Sequence:")

    # Explicit formula: a_n = 3n + 1
    arithmetic_explicit_formula = lambda n: 3 * n + 1
    arithmetic_explicit_sequence = generate_explicit_sequence(10, arithmetic_explicit_formula)

    # Recursive formula: a_n = a_(n-1) + 3, a_0 = 1
    arithmetic_recursive_relation = lambda seq, n: seq[n-1] + 3
    arithmetic_recursive_sequence = generate_recursive_sequence(10, [1], arithmetic_recursive_relation)

    print(f"Explicit Sequence: {arithmetic_explicit_sequence}")
    print(f"Recursive Sequence: {arithmetic_recursive_sequence}")
    print(f"Sequences Match: {arithmetic_explicit_sequence == arithmetic_recursive_sequence}")
    print(f"Finite Sum: {finite_sum(arithmetic_explicit_sequence)}")

    # Example 3: Geometric Sequence
    print("\nGeometric Sequence:")

    # Explicit formula: a_n = 2 * 3^n
    geometric_explicit_formula = lambda n: 2 * (3**n)
    geometric_explicit_sequence = generate_explicit_sequence(10, geometric_explicit_formula)

    # Recursive formula: a_n = a_(n-1) * 3, a_0 = 2
    geometric_recursive_relation = lambda seq, n: seq[n-1] * 3
    geometric_recursive_sequence = generate_recursive_sequence(10, [2], geometric_recursive_relation)

    print(f"Explicit Sequence: {geometric_explicit_sequence}")
    print(f"Recursive Sequence: {geometric_recursive_sequence}")
    print(f"Sequences Match: {geometric_explicit_sequence == geometric_recursive_sequence}")
    print(f"Finite Sum: {finite_sum(geometric_explicit_sequence)}")

    # Harmonic Series
    print("\nHarmonic Series (Partial Sums):")
    n_terms = 10
    harmonic_formula = lambda n: 1 / (n + 1) # indexing begins at 0
    harmonic_sequence = [harmonic_formula(n) for n in range(20)]
    # Generate the harmonic sequence
    harmonic_sequence = [harmonic_formula(n) for n in range(n_terms)]

    # Compute the finite sum
    harmonic_sum = sum(harmonic_sequence)

    # Output results
    print(f"Harmonic Sequence (First {n_terms} terms): {harmonic_sequence}")
    print(f"Finite Sum of the Harmonic Series (First {n_terms} terms): {harmonic_sum:.6f}")

    # Gregory-Leibniz Series for Pi
    print("\nGregory-Leibniz Series (Approximation of Pi):")
    gregory_formula = lambda n: 4 * (-1)**n / (2 * n + 1)
    gregory_sequence = [gregory_formula(n) for n in range(10000)]
    pi_approximation = sum(gregory_sequence)
    print(f"Pi Approximation (Gregory-Leibniz): {pi_approximation:.6f}")

    # Zeta Function (ζ(2)) Approximation
    print("\nZeta Function (ζ(2)): Sum of 1/n^2:")
    zeta_formula = lambda n: 1 / (n + 1)**2
    zeta_sequence = [zeta_formula(n) for n in range(1000)]
    zeta_sum = sum(zeta_sequence)
    print(f"ζ(2) Approximation: {zeta_sum:.6f}")

    print("\n\n\n")
    # Recursion Relation Between Coefficients
     # Define a sample recurrence relation: a_n = -a_(n-1) / n
    recurrence_relation = lambda coef, n: -coef[n-1] / n
    initial_terms = [1]  # Start with a_0 = 1

    # Generate coefficients
    n_terms = 25
    coefficients = generate_coefficients(n_terms, initial_terms, recurrence_relation)

    # Print the table of index vs coefficient term
    print("Table of Index vs Coefficient Term:")
    print_coefficient_table(coefficients)

    # Scatter plot of sequence
    scatter_plot_sequence(coefficients)

    # Scatter plot of series
    scatter_plot_series(coefficients)









# In[35]:


"""Segment 2 - Taylor Series """

import numpy as np
import matplotlib.pyplot as plt

# Define the mathematical function
def f(x):
    A = 1.0       # Amplitude
    mu = 0.0      # Mean
    sigma = 1.0   # Standard deviation
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Derivative Estimators
def first_derivative(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

def second_derivative(f, x0, h):
    return (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / (h ** 2)

def third_derivative(f, x0, h):
    return (f(x0 + 2*h) - 2*f(x0 + h) + 2*f(x0 - h) - f(x0 - 2*h)) / (2 * h ** 3)

def fourth_derivative(f, x0, h):
    return (f(x0 + 2*h) - 4*f(x0 + h) + 6*f(x0) - 4*f(x0 - h) + f(x0 - 2*h)) / (h ** 4)

# Taylor Series Estimator
def taylor_series(f, x0, h, n_terms, x_range):
    """
    Estimate the Taylor series for a function.
    :param f: Function to approximate.
    :param x0: Point of expansion.
    :param h: Step size for derivative estimates.
    :param n_terms: Number of terms in the Taylor series.
    :param x_range: Range of x values for the approximation.
    :return: Array of Taylor series approximations.
    """
    # Precompute derivative values
    derivatives = [
        f(x0),  # f(x0)
        first_derivative(f, x0, h),  # f'(x0)
        second_derivative(f, x0, h),  # f''(x0)
        third_derivative(f, x0, h),  # f'''(x0)
        fourth_derivative(f, x0, h),  # f''''(x0)
    ]
    
    # Compute Taylor series approximation
    taylor_approximation = np.zeros_like(x_range)
    for n in range(n_terms):
        if n >= len(derivatives):
            break
        taylor_approximation += derivatives[n] * ((x_range - x0) ** n) / np.math.factorial(n)
    
    return taylor_approximation

# Main Program
if __name__ == "__main__":
    # Inputs
    x0 = float(input("Enter the point of expansion (x0): "))
    h = float(input("Enter the step size (h): "))
    n_terms = int(input("Enter the number of terms in the Taylor series: "))
    x_range = np.linspace(-5, 5, 500)  # Full range of the original function
    x_restricted = np.linspace(x0 - 2, x0 + 2, 200)  # Restricted range for Taylor series

    # Compute the Taylor series approximation
    taylor_approx = taylor_series(f, x0, h, n_terms, x_restricted)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, f(x_range), label="Original Function", color="blue", linewidth=2)
    plt.plot(x_restricted, taylor_approx, label=f"Taylor Approximation (n={n_terms})", color="red", linestyle="--")
    plt.scatter([x0], [f(x0)], color="black", label=f"Expansion Point (x0={x0})")
    plt.title("Taylor Series Approximation (Restricted Range)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()



# In[ ]:




