#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Constants
g = -9.81  # acceleration due to gravity (m/s^2)
b = 0.2    # drag coefficient (kg/s)
time_step = 0.01  # time step for numerical integration
total_time = 10  # Total simulation time (s)
m = 5

def simulate_free_fall(m,y0, v0y, drag_type="none"):
    """
    Simulate the free fall of an object with and without air drag.

    Parameters:
        y0: Initial height (m)
        v0y: Initial velocity (m/s)
        drag_type: "none", "linear", or "quadratic"
    """
    num_steps = int(total_time / time_step)
    y = np.zeros(num_steps)
    v = np.zeros(num_steps)
    t = np.linspace(0, total_time, num_steps)

    y[0] = y0
    v[0] = v0y

    for i in range(1, num_steps):
        if drag_type == "none":
            fy = m*g
        elif drag_type == "linear":
            fy = (m*g) - b * v[i - 1]
        elif drag_type == "quadratic":
            fy = (m*g) - b * v[i - 1] * abs(v[i - 1])
        else:
            raise ValueError("Unknown drag type")

        v[i] = v[i - 1] + (fy/m) * time_step
        y[i] = y[i - 1] + v[i] * time_step

        # Stop if object hits the ground
        if y[i] < 0:
            y[i] = 0
            v[i] = 0
            break

    return t, y

# Initial conditions
y0 = 100  # Initial height (m)
v0y = 0   # Initial velocity (m/s)

# Simulate free fall for all drag types
drag_types = ["none", "linear", "quadratic"]
colors = {"none": "blue", "linear": "green", "quadratic": "red"}
results = {drag: simulate_free_fall(m,y0, v0y, drag) for drag in drag_types}

# Create the animation
fig, ax = plt.subplots(figsize=(6, 10))
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-10, y0 + 10)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.fill_between([-1, 3], -10, 0, color="gray", alpha=0.5)
ax.set_title("Free Fall Simulation", fontsize=14)
ax.set_ylabel("Height (m)")
ax.set_xlabel("Drag Condition")
ax.set_xticks(range(len(drag_types)))
ax.set_xticklabels(["No Drag", "Linear Drag", "Quadratic Drag"])

# Points for each drag type
points = {drag: ax.plot([], [], 'o', color=colors[drag], label=drag.capitalize())[0] for drag in drag_types}
ax.legend()

def init():
    for point in points.values():
        point.set_data([], [])
    return points.values()

def update(frame):
    for drag, (t, y) in results.items():
        if frame < len(t):
            points[drag].set_data([drag_types.index(drag)], [y[frame]])
    return points.values()

ani = FuncAnimation(fig, update, frames=int(total_time / time_step), init_func=init, interval=20, blit=True)
plt.close(fig)  # Prevents duplicate plot in notebooks

HTML(ani.to_jshtml())


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Constants
g = -9.81  # Acceleration due to gravity (m/s²)
b_linear = 0.01    # Drag coefficient (kg/s)
m = 2      # Mass (kg)
time_step = 0.01  # Time step for numerical integration (s)
total_time = 100  # Total simulation time (s)
y0 = 100   # Initial height (m)
v0y = 0    # Initial velocity (m/s)
drag_types = ["none", "linear", "quadratic"]
colors = {"none": "blue", "linear": "green", "quadratic": "red"}

# Time array
times = np.linspace(0, total_time, int(total_time / time_step))

# Initialize dictionaries to store results
accelerations = {}
velocities = {}
positions = {}

# Terminal velocities
vt_linear = m * abs(g) / b  # Terminal velocity for linear drag
vt_quadratic = np.sqrt(m * abs(g) / b)  # Terminal velocity for quadratic drag

# Run the simulation for each drag type
for drag in drag_types:
    num_steps = len(times)
    a = np.zeros(num_steps)
    v = np.zeros(num_steps)
    y = np.zeros(num_steps)

    y[0] = y0
    v[0] = v0y

    for i in range(1, num_steps):
        if drag == "none":
            a[i] = g
        elif drag == "linear":
            a[i] = g - (b / m) * v[i - 1]
        elif drag == "quadratic":
            a[i] = g - (b / m) * v[i - 1] * abs(v[i - 1])

        v[i] = v[i - 1] + a[i] * time_step
        y[i] = y[i - 1] + v[i] * time_step

    accelerations[drag] = a
    velocities[drag] = v
    positions[drag] = y

# Plot acceleration, velocity, and position
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot titles and labels
titles = ["Acceleration vs Time", "Velocity vs Time", "Position vs Time"]
ylabels = ["Acceleration (m/s²)", "Velocity (m/s)", "Position (m)"]

# Generate plots
for i, metric in enumerate([accelerations, velocities, positions]):
    for drag, data in metric.items():
        axs[i].plot(times, data, label=f"{drag.capitalize()} Drag", color=colors[drag])

    axs[i].set_title(titles[i])
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel(ylabels[i])

    # Add horizontal lines for terminal velocities on velocity plot
    if i == 1:  # Velocity plot
        axs[i].axhline(y=-vt_linear, color="green", linestyle="--", label="Linear Terminal Velocity")
        axs[i].axhline(y=-vt_quadratic, color="red", linestyle="--", label="Quadratic Terminal Velocity")

    # Configure ticks and grid
    axs[i].grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
    axs[i].tick_params(axis='both', which='major', labelsize=10, length=5, width=1)

# Add legends outside the plots
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)

# Adjust layout to ensure no overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams["animation.embed_limit"] = 50

# Constants
g = -9.81  # Acceleration due to gravity (m/s²)
b = 0.01  # Linear drag coefficient (kg/s)
time_step = 0.01  # Time step for numerical integration (s)
total_time = 30  # Total simulation time (s)
v0 = 10  # Initial velocity (m/s)
theta = 45  # Launch angle (degrees)

# Initial conditions
v0x = v0 * np.cos(np.radians(theta))  # Initial velocity in x-direction
v0y = v0 * np.sin(np.radians(theta))  # Initial velocity in y-direction
x0, y0 = 0, 0  # Initial position

# Time array
times = np.arange(0, total_time, time_step)

# Arrays to store positions and velocities
x_no_drag = np.zeros(len(times))
y_no_drag = np.zeros(len(times))
vx_no_drag = np.full(len(times), v0x)  # Horizontal velocity is constant
vy_no_drag = np.zeros(len(times))  # Vertical velocity array
x_linear_drag = np.zeros(len(times))
y_linear_drag = np.zeros(len(times))
vx_linear_drag = np.zeros(len(times))
vy_linear_drag = np.zeros(len(times))
x_quadratic_drag = np.zeros(len(times))
y_quadratic_drag = np.zeros(len(times))
vx_quadratic_drag = np.zeros(len(times))
vy_quadratic_drag = np.zeros(len(times))

# Initial conditions
x_no_drag[0], y_no_drag[0] = x0, y0
vy_no_drag[0] = v0y
x_linear_drag[0], y_linear_drag[0] = x0, y0
vx_linear_drag[0] = v0x
vy_linear_drag[0] = v0y
x_quadratic_drag[0], y_quadratic_drag[0] = x0, y0
vx_quadratic_drag[0] = v0x
vy_quadratic_drag[0] = v0y

# Calculate no-drag trajectory using arrays
for i in range(1, len(times)):
    # Acceleration is constant in the no-drag case
    ay_no_drag = g  # Vertical acceleration due to gravity

    # Update velocities
    vx_no_drag[i] = vx_no_drag[i - 1]  # No change in horizontal velocity
    vy_no_drag[i] = vy_no_drag[i - 1] + ay_no_drag * time_step  # Update vertical velocity

    # Update positions using velocities
    x_no_drag[i] = x_no_drag[i - 1] + vx_no_drag[i - 1] * time_step
    y_no_drag[i] = y_no_drag[i - 1] + vy_no_drag[i - 1] * time_step

    # Stop updating positions and velocities when the projectile reaches the ground
    if y_no_drag[i] < 0:
        y_no_drag[i] = 0
        x_no_drag[i] = x_no_drag[i - 1]
        vx_no_drag[i] = 0  # Stop horizontal velocity
        vy_no_drag[i] = 0  # Stop vertical velocity


# Calculate linear-drag trajectory
for i in range(1, len(times)):
    # Update velocities with linear drag
    ax_drag = -b * vx_linear_drag[i - 1]
    ay_drag = g - b * vy_linear_drag[i - 1]
    vx_linear_drag[i] = vx_linear_drag[i - 1] + ax_drag * time_step
    vy_linear_drag[i] = vy_linear_drag[i - 1] + ay_drag * time_step

    # Update positions
    x_linear_drag[i] = x_linear_drag[i - 1] + vx_linear_drag[i - 1] * time_step
    y_linear_drag[i] = y_linear_drag[i - 1] + vy_linear_drag[i - 1] * time_step

    # Stop updating positions when the projectile reaches the ground
    if y_linear_drag[i] < 0:
        vx_linear_drag[i] = 0
        vy_linear_drag[i] = 0
        y_linear_drag[i] = 0

# Calculate quadratic-drag trajectory
for i in range(1, len(times)):
    # Calculate speed and update velocities with quadratic drag
    speed_quadratic = np.sqrt(vx_quadratic_drag[i - 1]**2 + vy_quadratic_drag[i - 1]**2)
    ax_quadratic = -b * speed_quadratic * vx_quadratic_drag[i - 1]
    ay_quadratic = g - b * speed_quadratic * vy_quadratic_drag[i - 1]
    vx_quadratic_drag[i] = vx_quadratic_drag[i - 1] + ax_quadratic * time_step
    vy_quadratic_drag[i] = vy_quadratic_drag[i - 1] + ay_quadratic * time_step

    # Update positions
    x_quadratic_drag[i] = x_quadratic_drag[i - 1] + vx_quadratic_drag[i - 1] * time_step
    y_quadratic_drag[i] = y_quadratic_drag[i - 1] + vy_quadratic_drag[i - 1] * time_step

    # Stop updating positions when the projectile reaches the ground
    if y_quadratic_drag[i] < 0:
        vx_quadratic_drag[i] = 0
        vy_quadratic_drag[i] = 0
        y_quadratic_drag[i] = 0

# Animation setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, max(np.max(x_no_drag), np.max(x_linear_drag), np.max(x_quadratic_drag)) * 1.1)
ax.set_ylim(0, max(np.max(y_no_drag), np.max(y_linear_drag), np.max(y_quadratic_drag)) * 1.1)
ax.set_xlabel("Horizontal Position (m)")
ax.set_ylabel("Vertical Position (m)")
ax.set_title("Projectile Motion: No Drag, Linear Drag, and Quadratic Drag")

# Plot elements
point_no_drag, = ax.plot([], [], 'bo', label="No Air Drag")
trajectory_no_drag, = ax.plot([], [], 'b-')

point_linear_drag, = ax.plot([], [], 'ro', label="Linear Drag")
trajectory_linear_drag, = ax.plot([], [], 'r-')

point_quadratic_drag, = ax.plot([], [], 'go', label="Quadratic Drag")
trajectory_quadratic_drag, = ax.plot([], [], 'g-')

ax.legend()

# Update function for animation
def update(frame):
    if frame < len(x_no_drag):
        point_no_drag.set_data(x_no_drag[frame], y_no_drag[frame])
        trajectory_no_drag.set_data(x_no_drag[:frame + 1], y_no_drag[:frame + 1])
    if frame < len(x_linear_drag):
        point_linear_drag.set_data(x_linear_drag[frame], y_linear_drag[frame])
        trajectory_linear_drag.set_data(x_linear_drag[:frame + 1], y_linear_drag[:frame + 1])
    if frame < len(x_quadratic_drag):
        point_quadratic_drag.set_data(x_quadratic_drag[frame], y_quadratic_drag[frame])
        trajectory_quadratic_drag.set_data(x_quadratic_drag[:frame + 1], y_quadratic_drag[:frame + 1])
    return point_no_drag, trajectory_no_drag, point_linear_drag, trajectory_linear_drag, point_quadratic_drag, trajectory_quadratic_drag

# Create animation
ani = FuncAnimation(fig, update, frames=len(times), interval=20, blit=True)
plt.close(fig)  # Prevents static plot display in Jupyter Notebook

# Display animation
HTML(ani.to_jshtml())


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# Constants
g = -9.81  # Acceleration due to gravity (m/s²)
b = 0.1  # Drag coefficient (kg/s)
m = 1.0  # Mass of the projectile (kg)
v0 = 10  # Initial velocity (m/s)
time_step = 0.01  # Time step for numerical integration (s)
total_time = 30  # Total simulation time (s)
angles = np.linspace(0.1, 89.9, 500)  # Angles in degrees

# Function to calculate range numerically
def calculate_range(v0, theta, drag_type="none"):
    # Initial velocities
    v0x = v0 * np.cos(np.radians(theta))
    v0y = v0 * np.sin(np.radians(theta))

    # Arrays for positions and velocities
    x = 0
    y = 0
    vx = v0x
    vy = v0y

    range_x = 0  # To store the horizontal range

    for _ in np.arange(0, total_time, time_step):
        # Compute accelerations
        if drag_type == "none":
            ax = 0
            ay = g
        elif drag_type == "linear":
            ax = -b * vx / m
            ay = g - b * vy / m
        elif drag_type == "quadratic":
            speed = np.sqrt(vx**2 + vy**2)
            ax = -b * speed * vx / m
            ay = g - b * speed * vy / m

        # Update velocities
        vx += ax * time_step
        vy += ay * time_step

        # Update positions
        x += vx * time_step
        y += vy * time_step

        # Stop when the projectile hits the ground
        if y <= 0:
            range_x = x
            break

    return range_x

# Compute ranges for all angles
ranges_no_drag = []
ranges_linear_drag = []
ranges_quadratic_drag = []

for theta in angles:
    ranges_no_drag.append(calculate_range(v0, theta, drag_type="none"))
    ranges_linear_drag.append(calculate_range(v0, theta, drag_type="linear"))
    ranges_quadratic_drag.append(calculate_range(v0, theta, drag_type="quadratic"))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(angles, ranges_no_drag, label="No Drag", color="blue")
plt.plot(angles, ranges_linear_drag, label="Linear Drag", color="green")
plt.plot(angles, ranges_quadratic_drag, label="Quadratic Drag", color="red")
plt.title("Projectile Range vs Launch Angle")
plt.xlabel("Launch Angle (degrees)")
plt.ylabel("Range (m)")
plt.legend()
plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
plt.show()


# In[ ]:




