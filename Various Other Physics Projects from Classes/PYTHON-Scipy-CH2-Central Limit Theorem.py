#!/usr/bin/env python
# coding: utf-8

# In[43]:


# Diego Pantoja



import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm


# Uniform Distribution Test 
#par, cov = optimize.curve_fit(ExpGrowth,day_data1,subs_data1, p0 = [guess_x0, guess_r],sigma = unc, absolute_sigma=True )


Nexp1 = 1000000
Nbins1 = 60
Max1 = 3

r1 = np.random.uniform(low=-1.0, high = 1.0, size = Nexp1) # Fills an array of size Nexp1 with random values from -1 to +1 
h1,edges1 = np.histogram(r1,bins=Nbins1,range=(-Max1,Max1)) 
better_bins1 = (edges1[:-1] + edges1[1:])/2.0
plt.plot(edges1[:-1],h1,'b-', label = 'Uniform Dist.')
plt.title("Uniform Distribution Test")
plt.xlabel("$x$")
plt.ylabel("Entries")
plt.legend(loc='best')







# Gaussian Distribution Test

Nexp2 = 200000
Nbins2 = 60
Min2 = 0.0
Max2 = 10.0
Mean2 = 5.0
Sigma2 = 1.5


r2 = np.random.normal(loc = 5.0, scale = 1.5, size = Nexp2)
h2,edges2 = np.histogram(r2,bins=Nbins2,range=(Min2,Max2))
better_bins2 = (edges2[:-1] + edges2[1:])/2.0

plt.figure()
plt.plot(better_bins2,h2,'r--',label = 'Monte Carlo')

Binsize2 = (float(Max2 - Min2))/Nbins2
x2 = np.linspace(Min2,Max2,100)
y2 = Nexp2 * (Binsize2) * (stats.norm.pdf(x2,loc = Mean2,scale = Sigma2))
plt.plot(x2,y2,'k-', label = 'PDF')
plt.semilogy()
plt.title("Gaussian Distribution Test")
plt.xlabel("$x$")
plt.ylabel("Entries")
plt.legend(loc='best')








# PLOT 2.1





NEXP3 = 1000000

NAVG3 = 1

r31 = np.random.uniform(low=-1.0, high=1.0, size=(NEXP3,NAVG3))

Avg31 = np.sum(r31, axis=1)/float(NAVG3)

h3,edges3 = np.histogram(r31,bins=40,range=(-1.2,1.2))
better_bins3 = (edges3[:-1] + edges3[1:])/2.0

plt.figure()
plt.plot(better_bins3,h3,'r-',label = 'NAVG = 1')



NAVG3 = 2

r32 = np.random.uniform(low=-1.0, high=1.0, size=(NEXP3,NAVG3))

Avg32 = np.sum(r32, axis=1)/float(NAVG3)

h3,edges3 = np.histogram(r32,bins=40,range=(-1.2,1.2))
better_bins3 = (edges3[:-1] + edges3[1:])/2.0

plt.plot(better_bins3,h3,'g-',label = 'NAVG = 2')



NAVG3 = 3

r33 = np.random.uniform(low=-1.0, high=1.0, size=(NEXP3,NAVG3))

Avg33 = np.sum(r33, axis=1)/float(NAVG3)

h3,edges3 = np.histogram(r33,bins=40,range=(-1.2,1.2))
better_bins3 = (edges3[:-1] + edges3[1:])/2.0

plt.plot(better_bins3,h3,'b-',label = 'NAVG = 3')

plt.legend()
plt.xlabel("$x$")
plt.ylabel("Entries")
plt.title('2.1: Random Uniform Variable Plots')
plt.figure()


Mean31 = np.mean(Avg31)
Var31 = np.var(Avg31)
stdDev31 = pow(Var31,0.50)


Mean32 = np.mean(Avg32)
Var32 = np.var(Avg32)
stdDev32 = pow(Var32,0.50)



Mean33 = np.mean(Avg33)
Var33 = np.var(Avg33)
stdDev33 = pow(Var33,0.50)







h3,edges3 = np.histogram(Avg31,bins=40,range=(-1.2,1.2))
better_bins3 = (edges3[:-1] + edges3[1:])/2.0
plt.plot(better_bins3,h3,'r-',label = 'Gaussian NAVG = 1')

h3,edges3 = np.histogram(Avg32,bins=40,range=(-1.2,1.2))
better_bins3 = (edges3[:-1] + edges3[1:])/2.0
plt.plot(better_bins3,h3,'g-',label = 'Gaussian NAVG = 2')


h3,edges3 = np.histogram(Avg33,bins=40,range=(-1.2,1.2))
better_bins3 = (edges3[:-1] + edges3[1:])/2.0
plt.plot(better_bins3,h3,'b-',label = 'Gaussian NAVG = 3')
plt.legend(loc='best')
plt.xlabel("$x$")
plt.ylabel("Entries")
plt.title('2.2: Histograms Gaussian')











# Plot 2.2

NEXP4 = 1000000
NAVG4 = 10

r4 = np.random.uniform(low=-1.0, high=1.0, size=(NEXP4,NAVG4))

Avg4 = np.sum(r4, axis=1)/float(NAVG4)



Mean4 = np.mean(Avg4)



Var4 = np.var(Avg4)



stdDev4 = pow(Var4,0.50)



h4,edges4 = np.histogram(Avg4,bins=20,range=(-0.5,0.5))
better_bins4 = (edges4[:-1] + edges4[1:])/2.0


Binsize4 = (float(0.5 - (-0.5))/20)  



x4 = np.linspace(-0.50,0.50,100)
y4 = (NEXP4) * (Binsize4) * (stats.norm.pdf(x4,loc = Mean4,scale = stdDev4))

plt.figure()
plt.plot(better_bins4,h4,'r-',label = 'Data')
plt.plot(x4,y4,'k--', label = 'PDF')
plt.legend(loc='best')
plt.xlabel("$x$")
plt.ylabel("Entries")
plt.title('2.2: Histograms Gaussian (Log Scale)')
plt.semilogy()










# Plot 2.3



Nexp5 = 1000000
Nbins5 = 50

MeanA = 20
SigmaA = 3

MeanB = 50
SigmaB = 1.0

MeanC = MeanB - MeanA
SigmaC = pow((pow(SigmaA,2) + pow(SigmaB,2)),0.5)



SimA = np.random.normal(loc = MeanA, scale = SigmaA, size = Nexp5)
SimB = np.random.normal(loc = MeanB, scale = SigmaB, size = Nexp5)
SimC = np.random.normal(loc = MeanC, scale = SigmaC, size = Nexp5)


hA,edgesA = np.histogram(SimA,bins=Nbins5,range=((MeanA - 4*(SigmaA)),(MeanA + 4*(SigmaA))))
hB,edgesB = np.histogram(SimB,bins=Nbins5,range=((MeanB - 4*(SigmaB)),(MeanB + 4*(SigmaB))))
hC,edgesC = np.histogram(SimC,bins=Nbins5,range=(10,50))

better_binsA = (edgesA[:-1] + edgesA[1:])/2.0
better_binsB = (edgesB[:-1] + edgesB[1:])/2.0
better_binsC = (edgesC[:-1] + edgesC[1:])/2.0

plt.figure()
plt.plot(better_binsA,hA,'b-', label = 'a')
plt.plot(better_binsB,hB,'r-', label = 'b')
plt.plot(better_binsC,hC,'g--', label = 'c = b - a')
plt.legend(loc='best')
plt.xlabel("Quantity")
plt.ylabel("Entries")
plt.title('2.3: Histograms Gaussian')
plt.figure()










# Plot 2.4



Nexp5 = 1000000
Nbins5 = 50

MeanA = 25
SigmaA = 3

MeanB = 7.5
SigmaB = 1.0

MeanC = MeanA / MeanB
SigmaC = MeanC * pow(pow((SigmaA / MeanA), 2) + pow((SigmaB / MeanB) , 2),0.5 )


SimA = np.random.normal(loc = MeanA, scale = SigmaA, size = Nexp5)
SimB = np.random.normal(loc = MeanB, scale = SigmaB, size = Nexp5)
SimC = np.random.normal(loc = MeanA / MeanB, scale = SigmaC, size = Nexp5)



hA,edgesA = np.histogram(SimA,bins=Nbins5,range=((MeanA - 4*(SigmaA)),(MeanA + 4*(SigmaA))))
hB,edgesB = np.histogram(SimB,bins=Nbins5,range=((MeanB - 4*(SigmaB)),(MeanB + 4*(SigmaB))))
hC,edgesC = np.histogram(SimC,bins=Nbins5,range=((MeanC - 4*(SigmaC)),(MeanC + 4*(SigmaC))))

better_binsA = (edgesA[:-1] + edgesA[1:])/2.0
better_binsB = (edgesB[:-1] + edgesB[1:])/2.0
better_binsC = (edgesC[:-1] + edgesC[1:])/2.0

plt.figure()
plt.plot(better_binsA,hA,'b-', label = 'a')
plt.plot(better_binsB,hB,'r-', label = 'b')
plt.plot(better_binsC,hC,'g--', label = 'c = b / a')
plt.legend(loc='best')
plt.xlabel("Quantity")
plt.ylabel("Entries")
plt.title('2.4: Propagation of Uncertainties Division')
plt.figure()







# In[28]:


import pandas as pd
import numpy as np

series = pd.Series([3,5,6])

print(series)

series[0] = 4

print(series)

series['one'] = 5

print(series)




dict0 = { 'Key0': 0, 'Key1': 1 }
dict1 = { 'Key0': 2, 'Key1': 3 }

list_dicts = [dict0, dict1]

x = pd.DataFrame(list_dicts)

x5 = np.sin(x)

x6 = np.power(x,2)

x7 = np.log(x)

x8=np.log2(x)

x9 = np.expm1(x)

x10 = np.log1p(x)

x11 = np.exp(x)

x12 = x.abs()

x13 = abs(x)


print(x)


s1 = pd.Series([3,5,6])

s2 = pd.Series([3,5,6,7])

s3 = s1 + s2


print(s3)

sum1 = s3.sum()

sum12 = np.sum(s3)

print(sum1)

print(sum12)



df = pd.DataFrame(np.random.randint(1,11,(7, 5)),
                  columns=['a', 'b', 'c', 'd', 'e'])



df['f'] = df['a'] + df['b']

print(df)

new_list = df.iloc[5,2:5] + df.iloc[6,2:5]


df = df.append(new_list, ignore_index = True)


df['g'] = df.iloc[2:5,0] + df.iloc[2:5,1]

print(new_col)






print(df)


s11 = pd.Series([.2, .0, .6, .2])
s21 = pd.Series([.2, .0, .6, .2])
s11.corr(s21)


print(df.iloc[1,:].mean())







# In[155]:


import pandas as pd
import numpy as np


df = pd.read_csv("data.csv")

print(df)

index_list = df.iloc[:,0]

print(index_list)

df = pd.read_csv("data.csv", usecols =['a','b','c','d'], dtype={'a': float,'b': float, 'c': float,'d': float})

print(df)

df.set_index(index_list.values, inplace = True, append = True, drop = False)


#df = df.mask((df > 3))

series = pd.Series([3,5,6])

df = df[df > 3]

series = series[series < 6]


print(series)

print(df)


#df = df[df.mask(df.isnull())] 

#print(df)

print(df[df['a'].notnull()])




data = {('California', 2000): 33871648,
                ('California', 2010): 37253956,
                ('Texas', 2000): 20851820,
                ('Texas', 2010): 25145561,
                ('New York', 2000): 18976457,
                ('New York', 2010): 19378102}


pd.Series(data)


x3 = np.array([5,3,2,1])

arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue'], x3]

pd.MultiIndex.from_arrays(arrays, names=('number', 'color', 'number'))


pd.MultiIndex.from_tuples([('a', 1,5), ('a', 2,6), ('b', 1,3), ('b', 2,5)], names = (0,1,2))


df = pd.DataFrame([["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]], columns=["first", "second"])

print(df)



mi = pd.MultiIndex(levels=[['a', 'b'], [1, 2]], codes =[[0, 0, 1, 1], [0, 1, 0, 1]], names = ['x','y'])


dff = pd.DataFrame(index = mi)

print(dff)

dff + 5


# In[9]:


import numpy as np
import pandas as pd


# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

health_data = pd.DataFrame(data, index=index, columns=columns)
health_data


health_data.loc[2013,'Bob']


p = health_data.loc[pd.IndexSlice[2013, 1 , :]]

p['Bob']


q = health_data.loc[2013,'Bob']

w = q.loc[1,'HR']


d = (health_data.loc[2013,'Bob']).loc[1,'HR']

print(d)

print(q)

print(w)


print(health_data)


h = health_data.loc[:2013, 'Bob':'Guido']

print(h)



idx = pd.IndexSlice
h1 = h.loc[idx[:,[1]],:]


print(h1.loc[:, idx[:,['HR']]])


idx_rows = pd.IndexSlice
idx_cols = pd.IndexSlice

print(health_data.loc[idx_rows[[2013],[1]],idx_cols[['Bob', 'Guido'],['HR']]])


health_data.swaplevel('year','visit')


health_data.unstack(level=0)


array =[[1, 1, 3], ['Sharon', 'Nick', 'Bailey'],  
          ['Doctor', 'Scientist', 'Physicist']] 


midx = pd.MultiIndex.from_arrays(array)

print(midx)

print(pd.DataFrame(index = midx))

list0 = [0,0,1]

list1 = []

mix = pd.MultiIndex(levels = [[0,0,1], [5,4,3]], codes = [list0, list1])

                            

                                 
                                 


# In[18]:


import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.random.randint(1,11,(8, 5)),
                  columns=['a', 'b', 'c', 'd', 'e'])



print(df1)


df2 = pd.DataFrame(np.random.randint(1,11,(7, 2)),
                  columns=['g','h'])


print(df2)

df3 = pd.concat([df1,df2], axis = 1 )

print(df3)

print(df3['a'].sum())


# In[27]:


import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'data1': range(6),
                           'data2': rng.randint(0, 10, 6)},
                           columns = ['key', 'data1', 'data2'])

print(df.groupby('key').aggregate(['min',max,np.median,sum]))



def filter_func(x):
    return x['data2'].std() > 4

print(df);

print(df.groupby('key').filter(filter_func))

print(df.groupby('key').transform(lambda x: x - x.mean()))

print(df.groupby('key'))


# In[46]:


import re

test_string = 'abxcdefghijklmnopabcdefghi'

pattern = 'ab*c'

y = re.findall(pattern,test_string)

print(y)


# In[ ]:





# In[ ]:




