#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Diego Pantoja

#PHY 116: Python Excercises #1

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy import optimize
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm


#1.2 Binomial Process

print('1.2: Binomial Process')


# To simulate a Binomial process, we must have an event run with some probability of success/failure in each trial. 
# We run this random event ntry times and count the number of times a success occured. We can then record this number and repeat
# this experiment of ntry, nexp times recording the number of success we measured in each experient. The Binomial process has two
# Fundamental assumptions, 1 success and failure are mutally exclusive and 2 each trial is independent.

# To start our binomial process we want to fill an array with random values from a uniform distribution. Let's define the relevant variables

ntry = 5 # Number of trials/samples ran for each experiment
nexp = 3 # Number of experiments to run
eps = 0.5 # Prob of success

# x = np.random.uniform(0.0,1.0,10) This would fill a single array of random uniform values between 0,1 for 10 trials. But if we want to conduct multiple trials we will need a 2d array. With one dimension representing our trials and one dimension representing the experiments.


print('\n2d array of random values between 0 and 1')
x = np.random.uniform(0.0,1.0, size =(nexp, ntry))

print(x)

# This line of code will geneate random numbers into a 2d array of nexp rows and ntry columns.

# Row # = Experiment #

# Column # = The trial/sample #

# A specific entry in our matrix/2d array refers to a randomly generated result for a particular experiment and trial

# Next we incorporate the random trial/event into our code. With all values 0,1 being equally probable of being generated. To incorpate
# A random 2 outcome event with prob esp of occuring, we set a threshold value below which we will evaluate to true and above which will
# Evaluate to false.This can be done via a simple logic statement.

keep = (x <= eps)

# print(keep)

# This then generates an array of T/F, Success/Failures, testing each trial in each experiment (each entry in matrix), if it falls below the said threshold returning a T if it does.

bin_exp_results = keep.astype(int) # Converts True/False values in array of 0 and 1 values which we can mathematically work with.

print('\nTrial by Trial, results of simulation')
print(bin_exp_results)

success = np.zeros(nexp) # Fills array with zeros of size nexp

for i in range(0,nexp): # This line of code counts/sums the # of successes in each experiment/row
    sum = np.sum(bin_exp_results[i]) 
    success[i] = sum

print('\nNumber of Success Outcomes in each experiment')
print(success)

# From this point foward it becomes more convienent to use a function to which we can pass along our parameters and get an outcome.

def throw_binomial(nexp1,ntry1,eps1):
    x1 = np.random.uniform(size=(nexp1,ntry1))
    
    keep1 = ( x1 <= eps)
    y1 = keep1.astype(int)


    success1 = np.zeros(nexp1)


    
    for i in range(0,nexp1):
        sum = np.sum(y1[i]) 
        success1[i] = sum
        
    print('Mean Expected:      ', ntry1 * eps1) # Formulas from Analysis Ch 1.
    print('Mean Simulated:     ', np.mean(success1))
    print('Var Expected:       ', ntry1 * eps1 * (1-eps1))
    print('Var Simulated:      ', np.var(success1))
    
    return success1

print('\nBinomial Simulation: Function Simulation nexp = 10, ntry = 10')

eps1 = 0.5
nexp1 = 10
ntry1 = 10

y1 = throw_binomial(nexp1,ntry1,eps)

print('\nTest case with nexp = 100000:')

nexp2 = 100000

y2 = throw_binomial(nexp2,ntry1,eps1)


print('\nTest case with nexp = 100:, ntry = 100')

nexp3 = 100
ntry3 = 100

y2 = throw_binomial(nexp2,ntry1,eps1)





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




