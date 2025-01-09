#!/usr/bin/env python
# coding: utf-8

# In[70]:


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy import integrate

get_ipython().run_line_magic('matplotlib', 'inline')


# In[76]:


# EXERCISE 17.2

data = pd.read_csv('supernova_data.txt', sep='\s+') 
data


i = 0
x = []
y = []
y_err = []

while (i < 40):
    M = -19.3
    z = data['zcmb'][i]
    m = data['mb'][i]
    m_err = data['dmb'][i]

    D_A = ((pow(10,((m-M+5)/5))/(1+z)))/(1000000)
    DA_err = (D_A)*((0.2*2.302585093*m_err))
    
    x.append(z)
    y.append(D_A)
    y_err.append(DA_err)
    (i) = (i + 1)

plt.errorbar(x, y, yerr=y_err, linestyle='', marker='o')
plt.xlabel('z', size='large')
plt.ylabel('D_A (MPC)', size='large')
     


# In[72]:


# EXERCISE 17.3 PT 1

data = pd.read_csv('supernova_data.txt', sep='\s+') 
data

def integrand(x, H_0, V, m, k):
    return ((1/H_0)*pow((V + (m*(pow((1+x),3))) + (k*(pow((1+x),3)))),-0.5))

i = 0
x = []
y = []

while (i < 40):
    M = -19.3
    z = data['zcmb'][i]
    m = data['mb'][i]

    D_A = integrate.quad(integrand, 0, z, args = (73, 0.7, 0.3, 0.0 ))[0]
    
    x.append(z)
    y.append(D_A)

    (i) = (i + 1)

    
    
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.show()    

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.xlim(0.0, 0.2)
plt.show()    



# In[73]:


# EXERCISE 17.3 PT 2


data = pd.read_csv('supernova_data.txt', sep='\s+') 
data

def integrand(x, H_0, V, m, k):
    return ((1/H_0)*pow((V + (m*(pow((1+x),3))) + (k*(pow((1+x),3)))),-0.5))

i = 0
x = []
y = []

while (i < 40):
    M = -19.3
    z = data['zcmb'][i]
    m = data['mb'][i]

    D_A = integrate.quad(integrand, 0, z, args = (67, 0.7, 0.3, 0.0 ))[0]
    
    x.append(z)
    y.append(D_A)

    (i) = (i + 1)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.show()    

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.xlim(0.0, 0.2)
plt.show()    


# In[74]:


# EXERCISE 17.3 PT 3


data = pd.read_csv('supernova_data.txt', sep='\s+') 
data

def integrand(x, H_0, V, m, k):
    return ((1/H_0)*pow((V + (m*(pow((1+x),3))) + (k*(pow((1+x),3)))),-0.5))

i = 0
x = []
y = []

while (i < 40):
    M = -19.3
    z = data['zcmb'][i]
    m = data['mb'][i]

    D_A = integrate.quad(integrand, 0, z, args = (73, 0.0, 1.0, 0.0 ))[0]
    
    x.append(z)
    y.append(D_A)

    (i) = (i + 1)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.show()    

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.xlim(0.0, 0.2)
plt.show()    


# In[75]:


# EXERCISE 17.3 PT 4

data = pd.read_csv('supernova_data.txt', sep='\s+') 
data

def integrand(x, H_0, V, m, k):
    return ((1/H_0)*pow((V + (m*(pow((1+x),3))) + (k*(pow((1+x),3)))),-0.5))

i = 0
x = []
y = []

while (i < 40):
    M = -19.3
    z = data['zcmb'][i]
    m = data['mb'][i]

    D_A = integrate.quad(integrand, 0, z, args = (73, 0.0, 0.3, 0.7 ))[0]
    
    x.append(z)
    y.append(D_A)

    (i) = (i + 1)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.show()    

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(x,y)
plt.xlabel('z', size='large')
plt.ylabel(' D_A', size='large')
plt.xlim(0.0, 0.2)
plt.show()    


# In[ ]:




