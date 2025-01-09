#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib.pyplot as plt

# Constants all in SI
pi = 3.141592654
h = 6.62607015e-34  
c = 3.000e8    
k = 1.38065e-23 
me = 9.1093837e-31
pe = 20.0
Z1 = 1.0
Z2 = 2.0
Z3 = 1.0
x1 = 3.941e-18
x2 = 8.715e-18
k_exp = 8.618e-5


def N2_N1F(T):
    
    A = (2*k*T*Z2/pe * Z1)
    B = (math.pow(((2*pi*me*k*T)/(h*h)), 1.5))
    C = (math.exp(-x1/(k * T)))
    
    N2_N1 = A * B * C
    
    return N2_N1

def N3_N2F(T):
    
    A = (2*k*T*Z3/pe * Z2)
    B = (math.pow(((2*pi*me*k*T)/(h*h)), 1.5))
    C = (math.exp(-x2/(k * T)))
    
    N3_N2 = A * B * C
    
    return N3_N2


# Temperatures
Temperatures = np.array([5000.0,15000.0,25000.0])  


# Plotting
plt.figure(figsize=(10, 6))
for T in Temperatures:
    N2_N1 = N2_N1F(T)
    N3_N2 = N3_N2F(T)
    
    N2_NTOT = (N2_N1 / ( 1.0 + N2_N1 + (N2_N1)*(N3_N2)  ))
    
    print('Temperature (K)', T)
    print('N2 / N1 = ' ,N2_N1)
    print('N3 / N1 = ' ,N3_N2)
    print('N2 / NTOT = ' ,N2_NTOT)
    


Temp = np.arange(5000,35001,10)
N2_NTOT_array = []

for T in Temp:
    N2_N1 = N2_N1F(T)
    N3_N2 = N3_N2F(T)
    
    N2_NTOT = (N2_N1 / ( 1.0 + N2_N1 + (N2_N1)*(N3_N2)  ))
    N2_NTOT_array.append(N2_NTOT)
    
    
plt.figure(figsize=(10, 6))
plt.plot(Temp,N2_NTOT_array, "r")
plt.title('HW Number 4')
plt.xlabel('Temperature (K)')
plt.ylabel('NII / NTOT')
plt.axhline(y = 1/2, color = 'b', linestyle = 'dashed', label = 'Half Ionization') 
plt.legend(loc="upper left")
plt.grid(True)
plt.show()









# In[ ]:




