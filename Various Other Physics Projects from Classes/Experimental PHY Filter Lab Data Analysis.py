#!/usr/bin/env python
# coding: utf-8

# In[21]:


# LAB 6: FILTER LAB 

# Diego Pantoja Sharon Warmerdam


import numpy as np
import matplotlib.pyplot as plt
import math 

#Plot 6.1

# gain for low pass filter:


voltage_first = np.array([5.12,4.56,3.60,2.26,0.520])
voltage_second = np.array([8.40,7.92,5.92,1.92,0.092])
voltage_third = np.array([10.4,10.4,7.04,1.24,0.021])

g_first = voltage_first/5.12
g_second = voltage_second/8.40
g_third = voltage_third/10.4


freq = np.array([1.08, 5.4, 10.8, 21.6, 108])
freq_ratio = freq/10.8


# plot for expected response of low pass

def g_first_expected(arg1):
   numerator = 1.0
   t = pow(arg1,2.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   return g;


def g_second_expected(arg1):
   numerator = 1.0
   t = pow(arg1,4.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   return g;


def g_third_expected(arg1):
   numerator = 1.0
   t = pow(arg1,6.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   return g;



upper1 = 11
step1 = 0.001
x1 = np.arange(0.1, upper1, step)
y1 = g_first_expected(x1)



upper2 = 11
step2 = 0.001
x2 = np.arange(0.1, upper2, step)
y2 = g_second_expected(x1)

upper3 = 11
step2 = 0.001
x3 = np.arange(0.1, upper3, step)
y3 = g_third_expected(x1)



# Plot 6.3

# plot for expected response of low pass dB

def g_firstDB_expected(arg3):
   numerator = 1.0
   t = pow(arg3,2.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   dB1 = np.log10(g)
   return dB1

def g_secondDB_expected(arg3):
   numerator = 1.0
   t = pow(arg3,4.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   dB2 = np.log10(g)
   return dB2

def g_thirdDB_expected(arg3):
   numerator = 1.0
   t = pow(arg3,6.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   dB3 = np.log10(g)
   return dB3


   
y5 = 20* g_firstDB_expected(x1) # Theoretical Models
y6 = 20* g_secondDB_expected(x2)
y7 = 20* g_thirdDB_expected(x3)
dB_first = 20 * np.log10(g_first) 
dB_second = 20 * np.log10(g_second) 
dB_third = 20 * np.log10(g_third) 


plt.figure()
plt.plot(freq_ratio,dB_first,'ro',label = 'Data: 1st order')
plt.semilogx(x1,y5, color = 'red',label = 'Theoretical: 1st order')
plt.title('Gain(dB) vs Frequency')
plt.xlabel('f/$f_0$ (Log Scale)')
plt.ylabel('Gain (dB)')



plt.figure()
plt.plot(freq_ratio,dB_second,'bo', label = 'Data: 2nd order')
plt.semilogx(x2,y6, color = 'blue',label = 'Theoretical: 2nd order')
plt.title('Gain(dB) vs Frequency')
plt.xlabel('f/$f_0$ (Log Scale)')
plt.ylabel('Gain (dB)')

plt.figure()
plt.plot(freq_ratio,dB_third,'go', label = 'Data: 3rd order')
plt.semilogx(x3,y7, color = 'green',label = 'Theoretical: 3rd order')
plt.title('Gain(dB) vs Frequency')
plt.xlabel('f/$f_0$ (Log Scale)')
plt.ylabel('Gain (dB)')


plt.figure()
plt.plot(freq_ratio,dB_first,'ro',label = 'Data: 1st order')
plt.plot(freq_ratio,dB_second,'bo', label = 'Data: 2nd order')
plt.plot(freq_ratio,dB_third,'go', label = 'Data: 3rd order')

plt.semilogx(x1,y5, color = 'red',label = 'Theoretical: 1st order')
plt.semilogx(x2,y6, color = 'blue',label = 'Theoretical: 2nd order')
plt.semilogx(x3,y7, color = 'green',label = 'Theoretical: 3rd order')

plt.title('Gain(dB) vs Frequency')
plt.xlabel('f/$f_0$ (Log Scale)')
plt.ylabel('Gain (dB)')
plt.legend(loc=6, prop={'size': 8})










# In[ ]:




