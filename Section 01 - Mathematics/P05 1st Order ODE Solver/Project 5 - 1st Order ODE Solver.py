#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# User-defined ODE1
def f(x, y):
    return x + y  # Example function dy/dx = x + y 
def f1(x, y): # User-defined ODE1
    return x**2 + y**2  # Example function dy/dx = x^2 + y^2 

# Euler's Method
def euler_method(f, x0, y0, x_end, n):
    """
    Solve ODE using Euler's Method.
    """
    h = (x_end - x0) / n
    x = x0
    y = y0
    xs = [x]
    ys = [y]
    for step in range(n):
        m_avg = f(x, y)
        y += h * m_avg
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Improved Euler's Method
def improved_euler_method(f, x0, y0, x_end, n):
    """
    Solve ODE using Improved Euler's Method (Heun's method).
    """
    h = (x_end - x0) / n
    x = x0
    y = y0
    xs = [x]
    ys = [y]
    for step in range(n):
        m1 = f(x, y)
        m2 = f(x + h, y + h * m1)
        m_avg = (m1 + m2) / 2
        y += h * m_avg
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# RK4 Method
def rk4_method(f, x0, y0, x_end, n):
    """
    Solve ODE using the RK4 Method.
    """
    h = (x_end - x0) / n
    x = x0
    y = y0
    xs = [x]
    ys = [y]
    for step in range(n):
        m1 = f(x, y)
        m2 = f(x + h / 2, y + h * m1 / 2)
        m3 = f(x + h / 2, y + h * m2 / 2)
        m4 = f(x + h, y + h * m3)
        m_avg = (m1 + 2 * m2 + 2 * m3 + m4) / 6
        y += h * m_avg
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Plotting Results Without Points
def plot_results(xs, ys, method_name):
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label=method_name, linewidth=2)  # Removed 'marker="o"' for a smooth curve
    plt.title(f"{method_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

# Combined Plot for Comparison
def plot_combined(xs_euler, ys_euler, xs_improved, ys_improved, xs_rk4, ys_rk4):
    plt.figure(figsize=(12, 8))
    plt.plot(xs_euler, ys_euler, label="Euler's Method", linewidth=2, linestyle='--')
    plt.plot(xs_improved, ys_improved, label="Improved Euler's Method", linewidth=2, linestyle='-.')
    plt.plot(xs_rk4, ys_rk4, label="RK4 Method", linewidth=2)
    plt.title("Comparison of Numerical Methods")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

# Main Program
if __name__ == "__main__":
    # Inputs
    x0 = float(input("Enter initial x (x0): "))
    y0 = float(input("Enter initial y (y0): "))
    x_end = float(input("Enter final x (x_end): "))
    n = int(input("Enter the number of steps (n): "))

    # Euler's Method
    xs_euler, ys_euler = euler_method(f, x0, y0, x_end, n)
    plot_results(xs_euler, ys_euler, "Euler's Method")

    # Improved Euler's Method
    xs_improved, ys_improved = improved_euler_method(f, x0, y0, x_end, n)
    plot_results(xs_improved, ys_improved, "Improved Euler's Method")

    # RK4 Method
    xs_rk4, ys_rk4 = rk4_method(f, x0, y0, x_end, n)
    plot_results(xs_rk4, ys_rk4, "RK4 Method")

    plot_combined(xs_euler, ys_euler, xs_improved, ys_improved, xs_rk4, ys_rk4)


# In[21]:


import numpy as np
import matplotlib.pyplot as plt

# User-defined ODE1
def f(x, y):
    return x + y  # Example function dy/dx = x + y 
def f1(x, y): # User-defined ODE1
    return x**2 + y**2  # Example function dy/dx = x^2 + y^2 



# Improved Euler's Method
def improved_euler_method(f, x0, y0, x_end, n):
    """
    Solve ODE using Improved Euler's Method (Heun's method).
    """
    h = (x_end - x0) / n
    x = x0
    y = y0
    xs = [x]
    ys = [y]
    for step in range(n):
        m1 = f(x, y)
        m2 = f(x + h, y + h * m1)
        m_avg = (m1 + m2) / 2
        y += h * m_avg
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# RK4 Method
def rk4_method(f, x0, y0, x_end, n):
    """
    Solve ODE using the RK4 Method.
    """
    h = (x_end - x0) / n
    x = x0
    y = y0
    xs = [x]
    ys = [y]
    for step in range(n):
        m1 = f(x, y)
        m2 = f(x + h / 2, y + h * m1 / 2)
        m3 = f(x + h / 2, y + h * m2 / 2)
        m4 = f(x + h, y + h * m3)
        m_avg = (m1 + 2 * m2 + 2 * m3 + m4) / 6
        y += h * m_avg
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Plotting Results Without Points
def plot_results(xs, ys, method_name):
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label=method_name, linewidth=2)  # Removed 'marker="o"' for a smooth curve
    plt.title(f"{method_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

# Combined Plot for Comparison
def plot_combined(xs_euler, ys_euler, xs_improved, ys_improved, xs_rk4, ys_rk4):
    plt.figure(figsize=(12, 8))
    plt.plot(xs_euler, ys_euler, label="Euler's Method", linewidth=2, linestyle='--')
    plt.plot(xs_improved, ys_improved, label="Improved Euler's Method", linewidth=2, linestyle='-.')
    plt.plot(xs_rk4, ys_rk4, label="RK4 Method", linewidth=2)
    plt.title("Comparison of Numerical Methods")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

# Main Program
if __name__ == "__main__":
    # Inputs
    x0 = float(input("Enter initial x (x0): "))
    y0 = float(input("Enter initial y (y0): "))
    x_end = float(input("Enter final x (x_end): "))
    n = int(input("Enter the number of steps (n): "))


    # RK4 Method
    xs_rk4, ys_rk4 = rk4_method(f1, x0, y0, x_end, n)
    plot_results(xs_rk4, ys_rk4, "RK4 Method")



# In[ ]:




