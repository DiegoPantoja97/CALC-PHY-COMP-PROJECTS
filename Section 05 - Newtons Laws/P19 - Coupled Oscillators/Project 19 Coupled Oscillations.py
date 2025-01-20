#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from IPython.display import HTML

# Increase animation size limit
plt.rcParams["animation.embed_limit"] = 100  # Set limit to 100 MB

# Parameters
m1, m2 = 1.0, 1.0  # Masses
k1, k2, k3 = 2.0, 1.5, 2.0  # Spring constants
L1_eq, L2_eq, L3_eq = 0.75, 0.75, 0.75  # Natural lengths of the springs
t_max = 40  # Extended simulation time
dt = 0.01  # Time step

# Equilibrium positions
X1_eq = L1_eq  # Equilibrium position of the first mass
X2_eq = X1_eq + L2_eq  # Equilibrium position of the second mass
X3_eq = X2_eq + L3_eq  # Fixed position of the rightmost anchor

# Initial displacement (relative to equilibrium)
x1_0, x2_0 = 0.0, 0.5  # Small initial displacements
v1_0, v2_0 = 0.0, 0.0   # Initial velocities
y0 = [x1_0, x2_0, v1_0, v2_0]

# Define the system of ODEs
def coupled_odes(t, y):
    x1, x2, v1, v2 = y
    dx1dt = v1
    dx2dt = v2
    dv1dt = (-k1 * x1 + k2 * (x2 - x1)) / m1
    dv2dt = (-k3 * x2 + k2 * (x1 - x2)) / m2
    return [dx1dt, dx2dt, dv1dt, dv2dt]

# Solve the ODEs
t_eval = np.arange(0, t_max, dt)
sol = solve_ivp(coupled_odes, [0, t_max], y0, t_eval=t_eval, method='RK45')
x1_vals, x2_vals = sol.y[0], sol.y[1]

# Translate relative coordinates to absolute positions
X1_vals = x1_vals + X1_eq
X2_vals = x2_vals + X2_eq

# Animation setup
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlim(-0.5, X3_eq + 0.5)  # Adjust horizontal view range
ax.set_ylim(-0.3, 0.3)  # Adjust vertical view range for oscillations
ax.axhline(0, color='black', lw=1)  # Horizontal axis

# Plot equilibrium positions
ax.axvline(X1_eq, color='blue', lw=1, linestyle='--', label="Eq. Pos. Mass 1")
ax.axvline(X2_eq, color='red', lw=1, linestyle='--', label="Eq. Pos. Mass 2")

# Masses and springs
mass1, = ax.plot([], [], 'bo', markersize=20, label="Mass 1")
mass2, = ax.plot([], [], 'ro', markersize=20, label="Mass 2")
spring1, = ax.plot([], [], 'g-', lw=4, label="Spring 1")
spring2, = ax.plot([], [], 'k-', lw=4, label="Spring 2")
spring3, = ax.plot([], [], 'c-', lw=4, label="Spring 3")

# Initialize animation
def init():
    mass1.set_data([], [])
    mass2.set_data([], [])
    spring1.set_data([], [])
    spring2.set_data([], [])
    spring3.set_data([], [])
    return mass1, mass2, spring1, spring2, spring3

# Update function
def update(frame):
    X1 = X1_vals[frame]
    X2 = X2_vals[frame]
    
    # Update mass positions
    mass1.set_data(X1, 0)
    mass2.set_data(X2, 0)
    
    # Update spring positions
    spring1.set_data([0, X1], [0, 0])  # Fixed point to mass1
    spring2.set_data([X1, X2], [0, 0])  # Between mass1 and mass2
    spring3.set_data([X2, X3_eq], [0, 0])  # Mass2 to fixed point
    
    return mass1, mass2, spring1, spring2, spring3

# Create animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=dt*1000)

# Display animation in Jupyter Notebook
HTML(ani.to_jshtml())


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
m1, m2 = 1.5, 1.5  # Masses
k1, k2, k3 = 2.0, 2.0, 2.0  # Spring constants
L1_eq, L2_eq, L3_eq = 0.75, 0.75, 0.75  # Natural lengths of the springs
t_max = 40  # Extended simulation time
dt = 0.01  # Time step

# Equilibrium positions
X1_eq = L1_eq  # Equilibrium position of the first mass
X2_eq = X1_eq + L2_eq  # Equilibrium position of the second mass
X3_eq = X2_eq + L3_eq  # Fixed position of the rightmost anchor

# Initial displacement (relative to equilibrium)
x1_0, x2_0 = -0.2, 0.5  # Small initial displacements
v1_0, v2_0 = 0.0, 0.0   # Initial velocities
y0 = [x1_0, x2_0, v1_0, v2_0]

# Define the system of ODEs
def coupled_odes(t, y):
    x1, x2, v1, v2 = y
    dx1dt = v1
    dx2dt = v2
    dv1dt = (-k1 * x1 + k2 * (x2 - x1)) / m1
    dv2dt = (-k3 * x2 + k2 * (x1 - x2)) / m2
    return [dx1dt, dx2dt, dv1dt, dv2dt]

# Solve the ODEs
t_eval = np.arange(0, t_max, dt)
sol = solve_ivp(coupled_odes, [0, t_max], y0, t_eval=t_eval, method='RK45')
x1_vals, x2_vals = sol.y[0], sol.y[1]

# Plot x1 and x2 curves over time with a larger figure and legend outside the plot
plt.figure(figsize=(14, 8))  # Larger figure

# Plot the displacement curves
plt.plot(t_eval, x1_vals, label="x1 (Displacement of Mass 1)", color="blue", lw=2)
plt.plot(t_eval, x2_vals, label="x2 (Displacement of Mass 2)", color="red", lw=2, linestyle="-")

# Add labels, title, and grid
plt.title("Numerical Solution: Displacement of Masses vs. Time", fontsize=20)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Displacement (m)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)

# Add the legend outside the plot
plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:




