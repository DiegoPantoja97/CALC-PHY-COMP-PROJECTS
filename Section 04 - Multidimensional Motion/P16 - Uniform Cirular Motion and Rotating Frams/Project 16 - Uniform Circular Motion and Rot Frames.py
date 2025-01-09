#!/usr/bin/env python
# coding: utf-8

# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import rc

# Increase embedding limit to 50 MB
rc('animation', embed_limit=50)

# Inputs
r = 5.0       # Radius of the circle
omega = -2.0  # Angular speed (rad/s)
T = 2 * np.pi / abs(omega)  # Period of motion
num_points = 500            # Number of points for the simulation

# Time array
t = np.linspace(0, T, num_points)

# Coordinates of the particle
x = r * np.cos(omega * t)
y = r * np.sin(omega * t)

# Velocities and centripetal acceleration
vx = -r * omega * np.sin(omega * t)  # Velocity x-component
vy = r * omega * np.cos(omega * t)   # Velocity y-component
ac_x = -omega**2 * x  # Centripetal acceleration x-component
ac_y = -omega**2 * y  # Centripetal acceleration y-component

# Scaling factors for visualization
velocity_scale = 0.15  # Adjust as needed
acceleration_scale = 0.05  # Adjust as needed

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-r - 1, r + 1)
ax.set_ylim(-r - 1, r + 1)
ax.set_aspect('equal')
ax.set_title("Uniform Circular Motion with Velocity and Centripetal Acceleration", fontsize=16)
ax.set_xlabel("x (m)", fontsize=14)
ax.set_ylabel("y (m)", fontsize=14)

# Static circle and animated elements
circle = plt.Circle((0, 0), r, color="blue", fill=False, linewidth=2)
ax.add_artist(circle)
particle, = ax.plot([], [], 'ro')  # Particle
trajectory, = ax.plot([], [], 'g-', label="Trajectory")  # Trajectory line

# Initialize dynamic quivers with placeholders
velocity_vector = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1, color="orange", label="Velocity")
acceleration_vector = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1, color="purple", label="Centripetal Acceleration")

# Text to display coordinates
text_coordinates = ax.text(-r, -r - 2.5, "", fontsize=12, color="green")  # Offset downward

# Initialize the animation
def init():
    particle.set_data([], [])
    trajectory.set_data([], [])
    velocity_vector.set_UVC(0, 0)  # Reset dynamic vectors
    acceleration_vector.set_UVC(0, 0)
    text_coordinates.set_text("")
    return particle, trajectory, velocity_vector, acceleration_vector, text_coordinates

# Update function for animation
def update(frame):
    # Update particle position
    particle.set_data([x[frame]], [y[frame]])  # Single-element lists
    # Update trajectory line up to the current frame
    trajectory.set_data(x[:frame+1], y[:frame+1])
    # Update velocity vector at the particle's position
    velocity_vector.set_offsets([x[frame], y[frame]])  # Set origin to particle position
    velocity_vector.set_UVC(velocity_scale * vx[frame], velocity_scale * vy[frame])  # Set direction and magnitude
    # Update centripetal acceleration vector at the particle's position
    acceleration_vector.set_offsets([x[frame], y[frame]])  # Set origin to particle position
    acceleration_vector.set_UVC(acceleration_scale * ac_x[frame], acceleration_scale * ac_y[frame])  # Set direction and magnitude
    # Update coordinate text with offset
    text_coordinates.set_text(f"t={t[frame]:.2f}s: x={x[frame]:.2f}, y={y[frame]:.2f}")
    return particle, trajectory, velocity_vector, acceleration_vector, text_coordinates

# Create the animation
ani = FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=20)

# Display the animation in the notebook
animation_html = HTML(ani.to_jshtml())

# After the animation, plot x vs t and y vs t
def plot_position_vs_time():
    plt.figure(figsize=(10, 6))

    # Plot x vs t
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label="x(t)", color="blue")
    plt.title("Position vs Time")
    plt.ylabel("x (m)")
    plt.grid(True)
    plt.legend()

    # Plot y vs t
    plt.subplot(2, 1, 2)
    plt.plot(t, y, label="y(t)", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Show the animation
display(animation_html)

# Plot position vs time
plot_position_vs_time()


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Inputs
radius = 5.0      # Radius of the disk
omega = 2.0       # Angular velocity (rad/s)
duration = 5.0    # Total simulation time (seconds)
num_points = 500  # Number of points on the disk boundary for visualization
time_steps = 300  # Number of frames in the animation

# Time array
t = np.linspace(0, duration, time_steps)

# Disk Boundary Points
angles = np.linspace(0, 2 * np.pi, num_points)
disk_x = radius * np.cos(angles)  # Boundary x-coordinates
disk_y = radius * np.sin(angles)  # Boundary y-coordinates

# Internal Points of the Disk (to visualize motion inside the disk)
internal_points = np.random.uniform(-radius, radius, size=(100, 2))
internal_points = internal_points[np.linalg.norm(internal_points, axis=1) <= radius]

# Rotation Matrix Function
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-radius - 1, radius + 1)
ax.set_ylim(-radius - 1, radius + 1)
ax.set_aspect('equal')
ax.set_title("Rotating Disk", fontsize=16)
ax.set_xlabel("x (m)", fontsize=14)
ax.set_ylabel("y (m)", fontsize=14)

# Disk boundary and internal points
boundary_line, = ax.plot([], [], 'b-', label="Disk Boundary")  # Disk outline
internal_points_plot, = ax.plot([], [], 'ro', markersize=3, label="Internal Points")  # Internal points

# Initialize the animation
def init():
    boundary_line.set_data([], [])
    internal_points_plot.set_data([], [])
    return boundary_line, internal_points_plot

# Update function for animation
def update(frame):
    theta = omega * t[frame]  # Current rotation angle
    rot_matrix = rotation_matrix(theta)
    
    # Rotate boundary points
    rotated_boundary = rot_matrix @ np.vstack((disk_x, disk_y))
    boundary_line.set_data(rotated_boundary[0], rotated_boundary[1])
    
    # Rotate internal points
    rotated_internal = rot_matrix @ internal_points.T
    internal_points_plot.set_data(rotated_internal[0], rotated_internal[1])
    
    return boundary_line, internal_points_plot

# Create the animation
ani = FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=True, interval=20)

# Display the animation in the notebook
HTML(ani.to_jshtml())


# In[74]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Inputs
radius = 5.0      # Radius of the disk
omega = 0.5       # Angular velocity (rad/s)
duration = 5.0    # Total simulation time (seconds)
num_points = 500  # Number of points on the disk boundary for visualization
time_steps = 300  # Number of frames in the animation

# Time array
t = np.linspace(0, duration, time_steps)
dt = t[1] - t[0]  # Time step

# Disk Boundary Points
angles = np.linspace(0, 2 * np.pi, num_points)
disk_x = radius * np.cos(angles)  # Boundary x-coordinates
disk_y = radius * np.sin(angles)  # Boundary y-coordinates

# Internal Points of the Disk (to visualize motion inside the disk)
internal_points = np.random.uniform(-radius, radius, size=(100, 2))
internal_points = internal_points[np.linalg.norm(internal_points, axis=1) <= radius]

# Ball motion in inertial frame (straight-line outward motion)
r_initial = 0.0  # Initial radial position
v_initial = 1.0  # Initial radial velocity
r = r_initial + v_initial * t  # Radial position (straight-line outward motion)
x_inertial = r * np.cos(0)  # Straight-line motion along x-axis
y_inertial = r * np.sin(0)

# Transform ball motion to the rotating frame
x_rotating = x_inertial * np.cos(-omega * t) - y_inertial * np.sin(-omega * t)
y_rotating = x_inertial * np.sin(-omega * t) + y_inertial * np.cos(-omega * t)

# Rotation Matrix Function
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

# ---------------------- Inertial Frame Animation ----------------------
fig_inertial, ax_inertial = plt.subplots(figsize=(8, 8))
ax_inertial.set_xlim(-radius - 1, radius + 1)
ax_inertial.set_ylim(-radius - 1, radius + 1)
ax_inertial.set_aspect('equal')
ax_inertial.set_title("Inertial Frame", fontsize=16)
ax_inertial.set_xlabel("x (m)")
ax_inertial.set_ylabel("y (m)")

# Disk boundary, internal points, and ball
boundary_line, = ax_inertial.plot([], [], 'b-', label="Disk Boundary")  # Disk outline
internal_points_plot, = ax_inertial.plot([], [], 'ro', markersize=3, label="Internal Points")  # Internal points
ball_inertial, = ax_inertial.plot([], [], 'go', markersize=8, label="Ball (Inertial Frame)")  # Ball
ball_trajectory, = ax_inertial.plot([], [], 'g-', label="Ball Trajectory")  # Ball trajectory trace

# Initialize the animation
def init_inertial():
    boundary_line.set_data([], [])
    internal_points_plot.set_data([], [])
    ball_inertial.set_data([], [])
    ball_trajectory.set_data([], [])
    return boundary_line, internal_points_plot, ball_inertial, ball_trajectory

# Update function for animation
def update_inertial(frame):
    theta = omega * t[frame]  # Current rotation angle
    rot_matrix = rotation_matrix(theta)
    
    # Rotate boundary points
    rotated_boundary = rot_matrix @ np.vstack((disk_x, disk_y))
    boundary_line.set_data(rotated_boundary[0], rotated_boundary[1])
    
    # Rotate internal points
    rotated_internal = rot_matrix @ internal_points.T
    internal_points_plot.set_data(rotated_internal[0], rotated_internal[1])
    
    # Ball motion in inertial frame
    ball_inertial.set_data([x_inertial[frame]], [y_inertial[frame]])
    
    # Trace ball trajectory
    ball_trajectory.set_data(x_inertial[:frame+1], y_inertial[:frame+1])
    
    return boundary_line, internal_points_plot, ball_inertial, ball_trajectory

ani_inertial = FuncAnimation(fig_inertial, update_inertial, frames=time_steps, init_func=init_inertial, blit=True, interval=20)

# ---------------------- Rotating Frame Animation ----------------------
fig_rotating, ax_rotating = plt.subplots(figsize=(8, 8))
ax_rotating.set_xlim(-radius - 1, radius + 1)
ax_rotating.set_ylim(-radius - 1, radius + 1)
ax_rotating.set_aspect('equal')
ax_rotating.set_title("Rotating Frame", fontsize=16)
ax_rotating.set_xlabel("x' (m)")
ax_rotating.set_ylabel("y' (m)")

# Disk boundary and ball in rotating frame
disk_rotating, = ax_rotating.plot(disk_x, disk_y, 'b-', label="Stationary Disk (Rotating Frame)")
ball_path_rotating, = ax_rotating.plot([], [], 'g-', label="Ball Path (Rotating Frame)")  # Ball path

# Initialize the animation
def init_rotating():
    ball_path_rotating.set_data([], [])
    return ball_path_rotating,

# Update function for animation
def update_rotating(frame):
    ball_path_rotating.set_data(x_rotating[:frame], y_rotating[:frame])  # Ball path in rotating frame
    return ball_path_rotating,

ani_rotating = FuncAnimation(fig_rotating, update_rotating, frames=time_steps, init_func=init_rotating, blit=True, interval=20)

# ---------------------- Display Animations ----------------------
HTML("""
<div style="display: flex; flex-direction: row;">
    <div style="width: 50%;">""" + ani_inertial.to_jshtml() + """</div>
    <div style="width: 50%;">""" + ani_rotating.to_jshtml() + """</div>
</div>
""")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




