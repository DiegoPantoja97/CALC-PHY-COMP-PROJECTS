#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
import numpy as np
import math

def ODE1(k,m,t,x,p):
    return float(p/m)

def ODE2(k,m,t,x,p):
    return float(-k*x)

def ODESOLVE(k,m,t0,tf,x0,p0):   

    delta_t = (tf - t0)/10000
    
    
    x_range = np.array([1.0])
    p_range = np.array([3.2])
    
    x_range[0] = x0
    p_range[0] = p0
    
    x = x0
    p = p0
     
    
    for t in np.arange(t0,tf,delta_t):   
        
        x = x + (ODE1(k,m,t,x,p)*delta_t)
        p = p + (ODE2(k,m,t,x,p)*delta_t)
        x_range = np.append(x_range,x)
        p_range = np.append(p_range,p)
        
    plt.figure()
    plt.xlabel('Positon X (m)')
    plt.ylabel('Momentum P (kg m/s)')
    plt.plot(x_range,p_range,color = 'red',label = 'Numerical Solution')
    plt.legend(bbox_to_anchor=(0.4, 1.25), loc="upper right")
    
    plt.figure()
    plt.xlabel('Positon X (m)')
    plt.ylabel('Momentum P (kg m/s)')
    plt.plot(x_range,p_range,color = 'red',label = 'Numerical Solution')
    
    
    
    
def xt(x0,v0,k,m,t):
    w = math.sqrt(k/m)
    x = ((x0 * math.cos(w*t)) + ((v0/w) *math.sin(w*t)))
    return x
    
def pt(x0,v0,k,m,t):
    w = math.sqrt(k/m)
    p = m * (((-x0*w) * math.sin(w*t)) + ((v0) *math.cos(w*t)))
    return p
    
    
       
k = input("Enter the Spring/Stiffness Constant: ") 
m = input("Enter the Mass: ") 
x0 = input("Enter the initial position: ") 
p0 = input("Enter the initial momentum: ") 
t0 = input("Enter the initial time: ")
tf = input("Enter the Final time: ")
t0 = float(t0)
tf = float(tf)
k = float(k)
m = float(m)
x0 = float(x0)
p0 = float(p0)

v0 = p0 / m

ODESOLVE(k,m,t0,tf,x0,p0)

x_t = np.array([1.0])
p_t = np.array([3.1])
    
x_t[0] = x0
p_t[0] = p0

for t in np.arange(t0,tf,0.01):
    x = xt(x0,v0,k,m,t)
    p = pt(x0,v0,k,m,t)
    x_t = np.append(x_t,x)
    p_t = np.append(p_t,p)

plt.plot(x_t,p_t,color = 'blue', label = 'Explicit Solution')
plt.legend(bbox_to_anchor=(0.4, 1.25), loc="upper right")



q = input("Another Trajectory? 0 = No, 1 = Yes")
q = int(q)





while(q==1):
    k = input("Enter the Spring/Stiffness Constant:")
    m = input("Enter the Mass: ") 
    x0 = input("Enter the initial position: ") 
    p0 = input("Enter the initial momentum: ") 
    t0 = input("Enter the initial time: ")
    tf = input("Enter the Final time: ")
    t0 = float(t0)
    tf = float(tf)
    k = float(k)
    m = float(m)
    x0 = float(x0)
    p0 = float(p0)
    v0 = p0 / m
    ODESOLVE(k,m,t0,tf,x0,p0)
    
    x_t[0] = x0
    p_t[0] = p0

    for t in np.arange(t0,tf,delta_t):
        x = xt(x0,v0,k,m,t)
        p = pt(x0,v0,k,m,t)
        x_t = np.append(x_t,x)
        p_t = np.append(p_t,p)
        
    plt.plot(x_t,p_t,color = 'blue', label = 'Explicit Solution')
    plt.legend(bbox_to_anchor=(0.4, 1.25), loc="upper right")
        
    
    q = input("Another Trajectory? 0 = No, 1 = Yes")
    q = int(q)
    
    


    





  
   
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:




