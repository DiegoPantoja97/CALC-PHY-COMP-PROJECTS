#!/usr/bin/env python
# coding: utf-8

# In[59]:



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import rc

# Increase embedding limit
rc('animation', embed_limit=50)  # Set limit to 50 MB

# Constants
g = 9.81  # Gravity (m/s^2)

# Initial conditions
x0, y0 = 0, 10  # Initial position (y0 = 10 meters for general case)
v0 = 30         # Initial velocity magnitude (m/s)
angle_deg = 45  # Launch angle (degrees)

# Derived quantities
angle_rad = np.radians(angle_deg)
v0x = v0 * np.cos(angle_rad)
v0y = v0 * np.sin(angle_rad)

# General formula for time of flight
def calculate_time_of_flight(y0, v0y, g):
    a = 0.5 * g
    b = -v0y
    c = -y0
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real solution for time of flight (check initial conditions).")
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    return max(t1, t2)  # Return the positive root

# Compute time of flight
t_flight = calculate_time_of_flight(y0, v0y, g)
t = np.linspace(0, t_flight, 500)

# Trajectory equations
x = x0 + v0x * t
y = y0 + v0y * t - 0.5 * g * t**2
apex_y = (v0y**2) / (2 * g) + y0
xf = v0x * t_flight  # Final x-coordinate (horizontal distance)

# Animation setup
fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure size
ax.set_xlim(0, xf + 5)
ax.set_ylim(0, apex_y + 5)
ax.set_title("Projectile Motion Simulation", fontsize=18)
ax.set_xlabel("Horizontal Distance (m)", fontsize=14)
ax.set_ylabel("Vertical Distance (m)", fontsize=14)

# Mark initial position with explicit values
ax.scatter(x0, y0, color="blue", label=f"Initial Position ($x_0={x0}$, $y_0={y0}$)")
ax.text(x0 + 0.5, y0, f"($x_0={x0}$, $y_0={y0}$)", fontsize=12, color="blue", ha="left")

# Define plot elements
particle, = ax.plot([], [], 'ro', label="Projectile")
trajectory, = ax.plot([], [], 'b-', label="Path")

ax.legend(fontsize=12)
text_apex = ax.text(0, 0, "", color="red", fontsize=14)
text_final_xf = ax.text(0, 0, "", color="blue", fontsize=14)
text_time = ax.text(0, 0, "", color="green", fontsize=14)

def init():
    particle.set_data([], [])
    trajectory.set_data([], [])
    text_apex.set_text("")
    text_final_xf.set_text("")
    text_time.set_text("")
    return particle, trajectory, text_apex, text_final_xf, text_time

def update(frame):
    # Update particle position
    particle.set_data([x[frame]], [y[frame]])
    trajectory.set_data(x[:frame+1], y[:frame+1])
    
    # Annotations
    if frame == np.argmax(y):  # Apex
        text_apex.set_position((x[frame], y[frame]))
        text_apex.set_text(f"Apex: {y[frame]:.2f} m")
    if frame == len(t) - 1:  # End of flight
        text_final_xf.set_position((x[frame], 0))
        text_final_xf.set_text(f"Final Position ($x_f$): {x[frame]:.2f} m")
        text_time.set_position((x[frame]/2, 1))
        text_time.set_text(f"Time in Air: {t_flight:.2f} s")
    
    return particle, trajectory, text_apex, text_final_xf, text_time

# Global variable to keep animation object alive
global ani
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=10)  # Faster animation

# Embed animation in notebook
HTML(ani.to_jshtml())


# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Constants
g = 9.81  # acceleration due to gravity, in m/s^2

# Initial conditions
v0 = 50  # initial velocity in m/s
angles = [10,20,30,40, 45,50,60,70,80]  # launch angles in degrees

# Time array for the maximum time of flight plus some buffer time
t_max = max([2 * v0 * np.sin(np.radians(angle)) / g for angle in angles])
t = np.linspace(0, t_max * 1.2, num=300)  # 300 time points from 0 to t_max * 1.2

# Setting up the figure, axis, and plot elements to animate
fig, ax = plt.subplots(figsize=(8, 4))  # Reduced figure size
#ax.set_xlim((0, v0**2 * np.sin(2*np.radians(max(angles))) / g * 1.2))
ax.set_xlim(0,300)
ax.set_ylim((0, (v0**2 * np.sin(np.radians(max(angles)))**2) / (2 * g) * 1.2))
lines = [ax.plot([], [], lw=1, label=f'{angle}°')[0] for angle in angles]  # Thinner lines

# Initialize function: plot background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Animation function: this is called sequentially
def animate(i):
    for j, angle in enumerate(angles):
        angle_rad = np.radians(angle)
        x = v0 * np.cos(angle_rad) * t[:i]
        y = v0 * np.sin(angle_rad) * t[:i] - 0.5 * g * t[:i]**2
        lines[j].set_data(x, y)
    return lines

# Call the animator
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(t), interval=40, blit=True)  # Increased interval

# Add legend and labels
ax.legend()
ax.set_title('Projectile Motion for Different Launch Angles')
ax.set_xlabel('Horizontal Distance (m)')
ax.set_ylabel('Vertical Distance (m)')
ax.grid(True)

# Display the animation in the notebook
HTML(ani.to_jshtml())
#TIme in air and horrizontal velocity


# In[ ]:





# In[ ]:





# In[ ]:




