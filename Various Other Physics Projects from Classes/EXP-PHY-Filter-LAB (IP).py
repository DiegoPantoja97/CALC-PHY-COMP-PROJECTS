#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math 

#Plot 6.1

# gain for low pass filter:

g_lowpass = np.array([1.00, 1.00, 1.00, 0.94, 0.70, 0.30, 0.127, 0.059, 0.0146])

# gain for high pass filter:

g_highpass = np.array([0.12, 0.33, 0.72, 0.94, 0.99])

# frequencies for both:

freql = np.array([0.106, 0.354, 1.061, 3.537, 10.61, 31.83, 106.1, 318.3, 1061])
freql_ratio = freql/10.61

freqh = np.array([1.061,3.538,10.61,31.830,106.1])
freqh_ratio = freqh/10.61

# Phase Angle for low pass filter (degrees)

phase_lowpass = np.array([-0.382,-1.27,-6.12,-19.1,-45.5,-73.38,-77.85,-82.5,-91.5])

# Phase Angle for high pass filter (degrees)

phase_highpass = np.array([92,71.7,43.9,18.2,6.11])

pi = 3.141592654

phase_lowpass_radians = (pi/180.0) * phase_lowpass
phase_highpass_radians = (pi/180.0) * phase_highpass

# plot for expected response of low pass

def g_lowpass_expected(arg1):
   numerator = 1.0
   t = pow(arg1,2.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   return g;

# plot for expected response of high pass

def g_highpass_expected(arg2):
   numerator = 1.0
   t = pow(arg2,-2.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   return g;

upper = 110
step = 0.001
x1 = np.arange(0.01, upper, step)
y1 = g_lowpass_expected(x1)
y2 = g_highpass_expected(x1)

plt.figure()
plt.plot(freql_ratio,g_lowpass,'ro',label = 'Data: Low Pass Filter')
plt.plot(freqh_ratio,g_highpass,'bo', label = 'Data: High Pass Filter')
plt.semilogx(x1,y1, color = 'red',label = 'Theoretical: Low Pass Filter')
plt.semilogx(x1,y2, color = 'blue',label = 'Theoretical: High Pass Filter')
plt.title('Plot 1: Gain vs Frequency')
plt.xlabel('f/$f_0$')
plt.ylabel('Gain: $V_O$ / $V_S$')
plt.legend(loc=5, prop={'size': 7})

# PLOT 6.2

plt.figure()
x1 = np.arange(0.01, upper, step)
y3 = -1* np.arctan(x1) # Theoretical models for Low pass and high pass filter
y4 = np.arctan(1/x1) 
plt.semilogx(x1,y3,color = 'red',label = 'Theoretical: Low Pass Filter')
plt.semilogx(x1,y4,color = 'blue',label = 'Theoretical: High Pass Filter')
plt.plot(freql_ratio,phase_lowpass_radians,'ro',label = 'Data: Low Pass Filter')
plt.plot(freqh_ratio,phase_highpass_radians,'bo', label = 'Data: High Pass Filter')
plt.title('Plot 2: Phase Angle vs Frequency')
plt.xlabel('f/$f_0$ ')
plt.ylabel('Phase Angle (Radians)')
plt.legend(loc=1, prop={'size': 8})


# Plot 6.3

# plot for expected response of low pass dB

def g_lowpassDB_expected(arg3):
   numerator = 1.0
   t = pow(arg3,2.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   dB1 = np.log10(g)
   return dB1

# plot for expected response of high pass dB

def g_highpassDB_expected(arg4):
   numerator = 1.0
   t = pow(arg4,-2.0)
   denominator = pow((1 + t),0.5)
   g= numerator / denominator
   dB2 = np.log10(g)
   return dB2
   
y5 = 20* g_lowpassDB_expected(x1) # Theoretical Models
y6 = 20* g_highpassDB_expected(x1)
dB_Low = 20 * np.log10(g_lowpass) # Data --> dB
dB_High = 20 * np.log10(g_highpass)

plt.figure()
plt.plot(freql_ratio,dB_Low,'ro',label = 'Data: Low Pass Filter')
plt.plot(freqh_ratio,dB_High,'bo', label = 'Data: High Pass Filter')
plt.semilogx(x1,y5, color = 'red',label = 'Theoretical: Low Pass Filter')
plt.semilogx(x1,y6, color = 'blue',label = 'Theoretical: High Pass Filter')
plt.title('Plot 3: Gain(dB) vs Frequency')
plt.xlabel('f/$f_0$')
plt.ylabel('Gain (dB)')
plt.legend(loc=8, prop={'size': 8})







# In[ ]:





# In[ ]:




