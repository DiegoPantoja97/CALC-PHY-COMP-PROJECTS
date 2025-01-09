#!/usr/bin/env python
# coding: utf-8

# In[59]:


import matplotlib.pyplot as plt
import numpy as np

# Define the variables
A = 0.5  # Amplitude, 
omega_0 = 0.75  # Natural frequency, 
tau = 0.1  # 

# Define the frequency range for plotting
omega = np.linspace(0, 5, 700)  # omega is varied from 0 to 5 (arbitrary choice for visualization)

# Define the equation
mu = (1 - (A * omega**2 * (omega**2 - omega_0**2)) / ((omega**2 - omega_0**2)**2 + omega**2 * tau**2))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(omega, mu, color = 'red', label='Permeability - μ(ω)')
plt.axvspan(0.764, 1.041, color='red', alpha=0.3, label ='μ(ω) < 0')

# Label the axes
plt.xlabel(r'$\omega$ - Frequency')
plt.ylabel(r'$\mu$ - Permeability')
plt.title('Permeability vs Frequency: Array of Split Rings')
plt.axhline(y=0.5, color='black', linestyle='--')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# In[60]:


# Zooming in on the vertical axis by setting the y-axis limits
# Re-importing numpy as the previous cell execution failed
import numpy as np
import matplotlib.pyplot as plt

# Constants for the equation - assumed since not provided
omega_p = 1  # Plasma frequency, can be any constant, value chosen for representation purposes

# Define the omega (angular frequency) range over which we want to plot epsilon
omega = np.linspace(0.1, 5*omega_p, 400)  # avoiding division by zero by starting at 0.1

# Calculate epsilon for the range of omega values using the given relation
epsilon = 1 - omega_p**2 / omega**2




plt.figure(figsize=(10, 6))
plt.plot(omega, epsilon, label=' Permitivity - ε(ω)')
plt.axvspan(0.0, 1, color='blue', alpha=0.3, label ='ε(ω) < 0')
plt.xlabel(r'$\omega$ - Frequency')
plt.ylabel(r'$\epsilon$ - Permitivity')
plt.title('Permitivity vs Frequency: Grid of Wires')
plt.axhline(y=1, color='black', linestyle='--')
plt.ylim(-10, 2)  # setting the limits for y-axis to zoom in
plt.grid(True)
plt.legend()
plt.show()


# In[58]:


import matplotlib.pyplot as plt
import numpy as np
import math

# Define the variables
A = 0.5  # Amplitude, 
omega_0 = 0.75  # Natural frequency,
omega_p = 1
tau = 0.1  # 

# Define the frequency range for plotting
omega = np.linspace(0.6, 5.0, 700)  # omega is varied from 0 to 5 (arbitrary choice for visualization)

# Define the equation
epsilon = 1 - omega_p**2 / omega**2
mu = (1 - (A * omega**2 * (omega**2 - omega_0**2)) / ((omega**2 - omega_0**2)**2 + omega**2 * tau**2))

# Accounting for the fact that u is complex, to find n we take the sqrt of a negative number 

P = ((A*omega*omega*omega)/((omega**2 - omega_0**2)**2 + omega**2 * tau**2))
r = np.sqrt(((epsilon * mu)**2) + ((P**2)*(epsilon**2)))
theta = 0.5 * (np.arctan(P/mu))

# Taking the - of square root for every value of epsilon and mu that are simultaneously negative
for i in range(len(mu)):
    if mu[i] < 0 and epsilon[i] < 0:
        r[i] = r[i] * -1

n = r * np.cos(theta)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(omega, n, color = 'green', label='Refractive Index - n(ω)')
plt.axvspan(0.7641, 1.041, color='green', alpha=0.3, label ='n(ω) < 0') # Range found via graphical analysis

# Label the axes
plt.xlabel(r'$\omega$ - Frequency')
plt.ylabel('n - Index of Refraction')
plt.title('Index of Refraction vs Frequency: Array of Split Rings')
plt.axhline(y=0.5, color='black', linestyle='--')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()




# In[ ]:




