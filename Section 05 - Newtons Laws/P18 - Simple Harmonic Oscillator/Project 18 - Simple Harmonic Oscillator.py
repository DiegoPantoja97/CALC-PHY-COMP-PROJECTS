#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from IPython.display import HTML

def spring_ode(state, t, m, k):
    """
    Define the ODE for the spring system.
    state: [x, v], where x is position and v is velocity
    t: time
    m: mass (kg)
    k: spring constant (N/m)
    """
    x, v = state
    dxdt = v
    dvdt = -(k / m) * x
    return [dxdt, dvdt]

def simulate_shm_ode(x0, v0, m=1.0, k=5.0, t_max=10.0, dt=0.02):
    """
    Simulate the SHM system using ODE integration.
    :param x0: Initial displacement
    :param v0: Initial velocity
    :param m: Mass of the block (kg)
    :param k: Spring constant (N/m)
    :param t_max: Maximum simulation time (s)
    :param dt: Time step (s)
    :return: t_array, x_array, v_array (time, position, and velocity arrays)
    """
    t_array = np.arange(0, t_max, dt)
    init_state = [x0, v0]
    sol = odeint(spring_ode, init_state, t_array, args=(m, k))
    x_array = sol[:, 0]  # Extract positions
    v_array = sol[:, 1]  # Extract velocities
    a_array = -(k / m) * x_array  # Compute accelerations
    return t_array, x_array, v_array, a_array

def animate_quantity(t_array, y_array, y_label, title, color):
    """
    Create an animation for a single quantity (x, v, or a) evolving over time.
    :param t_array: Time array
    :param y_array: Array of the quantity to animate
    :param y_label: Label for the y-axis
    :param title: Title of the animation
    :param color: Color of the marker and line
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(t_array[0], t_array[-1])
    ax.set_ylim(1.2 * min(y_array), 1.2 * max(y_array))
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)  # Reference line for y=0

    # Initialize the line and marker
    line, = ax.plot([], [], lw=2, color=color)
    marker, = ax.plot([], [], 'o', color=color)

    def init():
        line.set_data([], [])
        marker.set_data([], [])
        return line, marker

    def update(frame):
        # Update the line and marker with the current data
        line.set_data(t_array[:frame], y_array[:frame])
        marker.set_data(t_array[frame], y_array[frame])
        return line, marker

    ani = FuncAnimation(fig, update, frames=len(t_array), init_func=init, blit=True, interval=20)
    return HTML(ani.to_jshtml())

def plot_spring(x0, v0):
    """
    Plot the static spring and vectors dynamically based on initial block position x0
    and initial velocity v0.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    plt.title("Static Plot: Horizontal Spring with Vectors")
    
    # Ground
    ax.axhspan(-2, 0, facecolor='lightgray')
    ax.axhline(0, color='k')
    
    # Block
    block_width, block_height = 0.4, 0.3
    block_bottom = 0.0
    x_center = x0
    block_left = x_center - block_width / 2
    block = plt.Rectangle((block_left, block_bottom), block_width, block_height, fc='blue', ec='black')
    ax.add_patch(block)

    # Vertical lines
    anchor_x = -4.0
    ax.axvline(anchor_x, color='black', linestyle='-', linewidth=1.5)
    ax.axvspan(-5, anchor_x, facecolor='gray', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)

    # Spring
    block_center_y = block_bottom + block_height / 2
    n_coils, amplitude = 5, 0.15
    Xs = np.linspace(anchor_x, block_left, 200)
    Ys = block_center_y + amplitude * np.sin(2 * np.pi * n_coils * (Xs - anchor_x) / (block_left - anchor_x))
    ax.plot(Xs, Ys, lw=3.5, color='red')

    # Displacement vector
    displacement_offset = -0.2
    ax.arrow(0, block_center_y + displacement_offset, x_center, 0, head_width=0.05, head_length=0.2, fc='green', ec='green', label="Displacement")
    
    # Restoring force vector
    force_scale = 0.5
    ax.arrow(x_center, block_center_y, -force_scale * x_center, 0, head_width=0.05, head_length=0.2, fc='orange', ec='orange', label="-kx")

    # Velocity vector
    velocity_offset = 0.4
    ax.arrow(x_center, block_bottom + block_height + velocity_offset, v0, 0, head_width=0.05, head_length=0.2, fc='purple', ec='purple', label="Velocity")

    # Legend
    ax.legend(loc="upper left")
    plt.show()

def animate_block_motion(t_array, x_array, v_array):
    """
    Animate the block's motion in SHM with a dynamic velocity vector.
    :param t_array: Time array
    :param x_array: Position array
    :param v_array: Velocity array
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    plt.title("Animation: Block Motion in SHM with Dynamic Velocity Vector")
    
    # Ground and dashed equilibrium line
    ax.axhspan(-1, 0, facecolor='lightgray')  # Shade everything below y=0
    ax.axhline(0, color='k')  # Ground line
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)  # Dashed equilibrium line
    
    # Block
    block_width, block_height = 0.4, 0.3
    block_bottom = 0.0
    block = plt.Rectangle((x_array[0] - block_width / 2, block_bottom), block_width, block_height, fc='blue', ec='black')
    ax.add_patch(block)

    # Velocity vector
    velocity_scale = 0.5  # Scale factor for velocity vector length
    velocity_vector = ax.arrow(
        x_array[0], block_bottom + block_height / 2,  # Start at block's center
        v_array[0] * velocity_scale, 0,              # Initial direction and length
        head_width=0.05, head_length=0.2, fc='purple', ec='purple', label="Velocity"
    )

    def update(frame):
        nonlocal velocity_vector

        # Update block position
        x_center = x_array[frame]
        block_left = x_center - block_width / 2
        block.set_xy((block_left, block_bottom))
        
        # Update velocity vector
        velocity_vector.remove()  # Remove the old vector
        velocity_vector = ax.arrow(
            x_center, block_bottom + block_height / 2,  # Start at the block's center
            v_array[frame] * velocity_scale, 0,         # Direction and scaled length
            head_width=0.05, head_length=0.2, fc='purple', ec='purple'
        )

        return block, velocity_vector

    ani = FuncAnimation(fig, update, frames=len(t_array), interval=20, blit=True)
    return HTML(ani.to_jshtml())

if __name__ == "__main__":
    x0 = 2.0  # Initial displacement
    v0 = -0.5  # Initial velocity
    m = 1.0    # Mass of the block (kg)
    k = 5.0    # Spring constant (N/m)
    t_max = 10.0  # Maximum simulation time (s)
    dt = 0.02     # Time step (s)

    # Static plot with spring and vectors
    plot_spring(x0, v0)

    # Simulate SHM and animate block's motion
    t_array, x_array, v_array, a_array  = simulate_shm_ode(x0, v0, m=m, k=k, t_max=t_max, dt=dt)
    animation_html = animate_block_motion(t_array, x_array, v_array)
    display(animation_html)

    # Animate position
    anim_x = animate_quantity(t_array, x_array, y_label="Position (x)", title="Position vs Time", color="blue")
    display(anim_x)

    # Animate velocity
    anim_v = animate_quantity(t_array, v_array, y_label="Velocity (v)", title="Velocity vs Time", color="green")
    display(anim_v)

    # Animate acceleration
    anim_a = animate_quantity(t_array, a_array, y_label="Acceleration (a)", title="Acceleration vs Time", color="red")
    display(anim_a)


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider

def position_function(t, x0, omega, phi):
    """
    General solution for position in SHM:
    x(t) = x0 * cos(omega * t + phi)
    :param t: Time array
    :param x0: Amplitude
    :param omega: Angular frequency
    :param phi: Phase shift
    :return: Position array
    """
    return x0 * np.cos(omega * t + phi)

def plot_position(x0=1.0, k=5.0, m=1.0, phi=0.0):
    """
    Plot the position function x(t) = x0 * cos(omega * t + phi) with fixed axis ranges
    and sliders for dynamic control of parameters.
    :param x0: Initial displacement (amplitude)
    :param k: Spring constant
    :param m: Mass
    :param phi: Phase shift
    """
    # Define the time array
    t = np.linspace(0, 11, 1000)  # Time range fixed from 0 to 11

    # Calculate angular frequency
    omega = np.sqrt(k / m)

    # Compute position
    x = position_function(t, x0, omega, phi)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, lw=2, label=rf"$x(t) = {x0}\cos({omega:.2f}t + {phi:.2f})$")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Equilibrium (x=0)")
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Position (x)", fontsize=12)
    plt.title("SHM: General Solution $x(t) = x_0 \cos(\omega t + \phi)$", fontsize=14)
    plt.grid(True)
    
    # Fixed axis ranges
    plt.xlim(0, 11)  # Fixed time axis range
    plt.ylim(-5, 5)  # Fixed position axis range
    
    # Legend outside the plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=12)
    plt.tight_layout()
    plt.show()

# Create interactive sliders
interactive_plot = interactive(
    plot_position,
    x0=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description="x₀"),
    k=FloatSlider(min=0.1, max=10.0, step=0.1, value=5.0, description="k"),
    m=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description="m"),
    phi=FloatSlider(min=0.0, max=2 * np.pi, step=0.1, value=0.0, description="ϕ"),
)

# Display the interactive plot
interactive_plot


# In[ ]:




