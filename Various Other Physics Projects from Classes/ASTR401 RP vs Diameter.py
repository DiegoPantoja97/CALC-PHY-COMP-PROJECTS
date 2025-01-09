#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt

# Constants all in SI
h = 6.62607015e-34  
c = 3.000e8    
k = 1.38065e-23 
me = 9.1093837e-31
pe = 20.0


def N2_N1(wavelength, diameter):
    
    rp = 1.22 * (wavelength / diameter) 
    
    return rp


def N3_N2(wavelength, diameter):
    
    rp = 1.22 * (wavelength / diameter) 
    
    
    return rp



# Temperatures
Temperatures = np.array([5000.0,15000.0,25000.0])  


# Plotting
plt.figure(figsize=(10, 6))
for T in Temperatures:
    RP = (res_power(L,diameters)) 
    #plt.plot(diameters, RP, label=f'{L*1e9} nm')  # Wavelength in nm

plt.title('Resolving Power vs Diameter')
plt.xlabel('Diameter (cm)')
plt.ylabel('RP (Radians)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))

diameters = np.arange(5,200, 0.1)

for L in wavelengths:
    RP = (res_power(L,diameters)) 
    plt.plot(diameters, RP, label=f'{L*1e9} nm')  # Wavelength in nm
    
plt.title('Resolving Power vs Diameter')
plt.xlabel('Diameter (cm)')
plt.ylabel('RP (Radians)')

plt.legend()
plt.grid(True)
plt.show()






# In[ ]:





# In[ ]:




