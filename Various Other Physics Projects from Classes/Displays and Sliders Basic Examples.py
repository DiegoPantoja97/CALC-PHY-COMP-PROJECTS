#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function #Widgets inline
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pandas



import numpy as np #Arrays
import matplotlib.pyplot as plt #Plotting and mathematics
import math
from matplotlib import animation, rc
from IPython.display import HTML

from scipy import stats # Statistical packages
from scipy import optimize
from scipy.stats import binom 
from scipy.stats import poisson
from scipy.stats import norm




def sine_function(x,k,A):
    return (A*np.sin(k*x))

interact(sine_function, x =10, k = 10, A = 10);

#Example of application

val1 = input("Enter the name: ") 
  
# print the type of input value, example of user input
print(val1) 
  
  
val2 = input("Enter the number: ") 

  
val2 = int(val2) 
print(val2) 


fig, ax = plt.subplots()

ax.set_xlim(( 0, 2))
ax.set_ylim((-2, 2))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=20, 
                               blit=True)


HTML(anim.to_jshtml())

rc('animation', html='jshtml')

anim





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




