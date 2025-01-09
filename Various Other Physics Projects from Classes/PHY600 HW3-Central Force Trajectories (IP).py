#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import math

def drdt(u):
    return float(u)

def dudt(r,k,L):
    return float((math.pow(r,-3)) * (k-L))

def dodt(r,L):
    return float((L/math.sqrt(r*r)))

def ODESOLVE(k0,L0,r0,u0):   

    delta_t = 0.001
    k = k0
    L = L0
    E = 0.2
    t_range = np.arange(0,10+delta_t,delta_t)
    r_range = np.array([1.0])
    u_range = np.array([3.2])
    E_range = np.array([3.2])
    theta = np.array([1.0])
    
    r_range[0] = r0
    u_range[0] = u0
    E_range[0] = 0.0
    theta[0] = 0.00
    
    u = u0
    r = r0
    angle = theta[0]
    
    
    sum = 0.0
    
    
    
    
     
    
    for t in np.arange(0,10,delta_t):  
        
        E = ((L/(2*pow(r,2))) - (k/pow(r,2)) + (1/2*(pow(drdt(u),2))))
        
        angle = angle + (dodt(r,L)*delta_t)
        u = u + (dudt(r,k,L)*delta_t)
        r = r + (drdt(u)*delta_t)
        
        E_range = np.append(E_range,t)  
        r_range = np.append(r_range,r)
        u_range = np.append(u_range,u)
        theta = np.append(theta,angle)
        
        
    x = np.multiply(r_range,np.cos(theta))
    y = np.multiply(r_range,np.sin(theta))
    
    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x,y,color = 'red',label = 'Trajectory')
    
    

    



q = int(input("Another Trajectory? 1 = Yes "))

while(q==1):
    k0 = float(input("K:"))
    L0 = float(input("L: "))
    r0 = float(input("r0: "))
    u0 = float(input("u0: "))
    
    ODESOLVE(k0,L0,r0,u0)
    
    q = int(input("Another Trajectory? 1 = Yes "))



    
    


    





  
   
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:




