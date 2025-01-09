#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# Constants
h = 6.62607015e-34  
c = 3.000e8    
k = 1.38065e-23    

def planck_radiation(wavelength, temperature):
    
    exponent = (h * c) / (wavelength * k * temperature)
    return (2 * h * c**2) / (wavelength**5 * (np.exp(exponent) - 1))

# Wavelength
wavelengths = np.linspace(1e-7, 3e-6, 500)  # 100 nm to 3000 nm

# Temperatures 
temperatures = [4000.0,5800.0,10000.0]

# Plotting
plt.figure(figsize=(10, 6))
for T in temperatures:
    intensity = planck_radiation(wavelengths, T)
    plt.plot(wavelengths * 1e9, intensity, label=f'{T} K')  # Wavelength in nm

plt.title('Planck Radiation Curves')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral Radiance')
plt.legend()
plt.grid(True)
plt.show()
print("You can see the Stefan-Boltzman Law at play, Intensity is much larger with increasing T as I = OT^4")

print("To see the finer details of the blackbody curves I plotted it for different Temperatures")

temperatures = [4000.0,4500.0,5000.0]

for T in temperatures:
    intensity = planck_radiation(wavelengths, T)
    plt.plot(wavelengths * 1e9, intensity, label=f'{T} K')  # Wavelength in nm

plt.title('Planck Radiation Curves')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral Radiance')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




