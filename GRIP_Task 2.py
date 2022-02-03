#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics Task 2

# In[2]:


#Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

#To ignore warnings
import warnings as wg
wg.filterwarnings("ignore")


# # Reading data from Iris dataset

# In[3]:


#Reading data from Iris dataset
df=pd.read_csv('Iris.csv')


# # Visualising data

# In[4]:


df.head()


# In[5]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df['Species'].unique()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


iris=pd.DataFrame(df)
iris_df= iris.drop(columns=['Species','Id'])
iris_df.head()


# # __________________________________________________________________________________________________________

# # Finding optimum number of clusters

# ## The Elbow Method

# In[15]:


#Calculating the within-cluster sum of square

within_cluster_sum_of_square=[]

clusters_range=range(1,15)
for k in clusters_range:
    km=KMeans(n_clusters=k)
    km=km.fit(iris_df)
    within_cluster_sum_of_square.append(km.inertia_)
    


# In[16]:


#Plotting the "Within-cluster sum of square" against cluster range

plt.plot(clusters_range,within_cluster_sum_of_square,'go--',color='green')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of square')
plt.grid()
plt.show()


# In[17]:


from sklearn.cluster import KMeans

model=KMeans(n_clusters=3,init='k-means++', max_iter=300, n_init=10, random_state=0)
predictions= model.fit_predict(iris_df)


# # Plotting cluster centers

# In[21]:


x=iris_df.iloc[:,[0,1,2,3]].values
plt.scatter(x[predictions==0,0],x[predictions==0,1],s=25,c='red',label='Iris-setosa')
plt.scatter(x[predictions==1,0],x[predictions==1,1],s=25,c='blue',label='Iris-versicolor')
plt.scatter(x[predictions==2,0],x[predictions==2,1],s=25,c='green',label='Iris-virginica')

#Plotting the cluster centers
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],s=100, c='yellow',label='Centroids')
plt.legend()
plt.grid()
plt.show()

