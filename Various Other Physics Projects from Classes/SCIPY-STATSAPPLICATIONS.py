#!/usr/bin/env python
# coding: utf-8

# In[17]:


#Diego Pantoja

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy import optimize
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm

# LINE FIT FUNCTION

def line_func(x, a, b): # Defines a function with 3 parameters, the x input array, slope and intercept as a function array
    return (a*x) + b





# PLOT 3.1


x = np.array([1.0,2.0,3.0,4.0,5.0,6.0])
y = np.array([15.9,23.6,33.9,39.7,45.0,32.4])
unc_y  = np.array([3.0,3.0,3.0,3.0,10.0,20.0])

# PLOTS DATA
plt.errorbar(x,y,yerr=unc_y,fmt = "ro",label = 'Data')

# Initial Parameter guess
guess_a = 1.0
guess_b = 1.0

par, cov = optimize.curve_fit(line_func,x,y, p0 = [guess_a, guess_b])

fit_a = par[0]
fit_b = par[1]
x_fit = np.linspace(1.0,6.0,100) 
y_fit = (fit_a*(x_fit)) + fit_b

# PLOTS CURVE FIT
plt.plot(x_fit,y_fit,'k--',label = 'Line Fit (Biased)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot 3.1 ')
plt.legend()
plt.figure()









# Plot 3.2



plt.errorbar(x,y,yerr=unc_y,fmt = "ro",label = 'Data')
par2, cov2 = optimize.curve_fit(line_func,x,y, p0 = [guess_a, guess_b],sigma = unc_y)

fit_a2 = par2[0]
fit_b2 = par2[1]


x_fit2 = np.linspace(1.0,6.0,100) 
y_fit2 = (fit_a2*(x_fit2)) + fit_b2


plt.plot(x_fit2,y_fit2,'k--',label = 'Line Fit (Non-Biased)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot 3.2 ')
plt.legend()
plt.figure()



# 3.3

def const_func(x,a):
    return a

x_data = np.arange(100)
y_data = np.random.normal(50.0,10.0, size = 100)
y_unc3 = np.linspace(0.0,99.0,100) 
for j in range(0,100): #  fills an array with uncertainties
    y_unc3[j] = 10

guess_a = 0.0
par, cov = optimize.curve_fit(const_func, x_data, y_data, p0 = [guess_a], sigma = y_unc3, absolute_sigma=True)

unc = np.sqrt(np.diag(cov))
fit_a = par[0]
unc_a = unc[0]

print("Excercise 3.3\n")

print("Uncertainty:\n ", unc_a)


# 3.4


print("Excercise 3.4\n")


guess_a = 0.0
par1, cov1 = optimize.curve_fit(const_func, x_data, y_data, p0 = [guess_a], absolute_sigma=True)

unc1 = np.sqrt(np.diag(cov1))
fit_a = par1[0]
unc_a1 = unc1[0]


print("Uncertainty:\n ", unc_a1)

print("\nComment: Since we are no longer passing along our uncertainties into curve_fit, curve_fit by default, will scale our chi_square sum by one. But in reality it should be scaled, by the associated sigma^2 of each data pt (in this case) ")










# PLOT 3.5


def sine_func(x, k, A): # Defines a function with 3 parameters, the x input array, slope and intercept as a function array
    return (A*np.sin(k*x))


x_sine_data = ([0,1,2,3,4,5,6,7,8,9,10])
y_sine_data = ([5.3,15.0,19.2,6.8,-9.7,-17.4,-20.5,2.1,15.7,18.5,8.6])
y_sine_unc = ([2,2,2,2,2,2,2,2,2,2,2])


guess_k = 1.0
guess_A = 1.0

par4, cov4 = optimize.curve_fit(sine_func,x_sine_data,y_sine_data, p0 = [guess_k, guess_A],sigma=y_sine_unc,absolute_sigma=True)

fit_k = par4[0]
fit_A = par4[1]
x_fit_sine = np.linspace(0.0,10.0,100) 
y_fit_sine = (fit_A*np.sin(fit_k*x_fit_sine))


plt.errorbar(x_sine_data,y_sine_data,yerr=y_sine_unc,fmt = "ro",label = 'Data')
plt.plot(x_fit_sine,y_fit_sine,'k--',label = 'Sine Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot 12.3 ')
plt.legend()
plt.figure()






print('\n\nPrint: 3.5 \n')
print('\nFitted Amplitude:')
print(fit_A)
print('\n\nFitted k:')
print(fit_k)
print('\n\nUncertainty in Fitted Values: k and A')
print(actual_unc)

sigma_y4 = np.sqrt(np.diag(cov4))

actual_unc4 = sigma_y4/pow(11,0.5)









# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




