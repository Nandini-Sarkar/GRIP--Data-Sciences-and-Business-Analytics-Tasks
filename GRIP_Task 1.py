#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics Task 1

# In[1]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading data from url

# In[4]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)


# # Exploring data

# In[5]:


print(data.shape)
data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.plot(kind='scatter',x='Hours',y='Scores');
plt.show()


# In[9]:


data.corr(method='pearson')


# In[10]:


data.corr(method='spearman')


# In[11]:


hours=data['Hours']
scores=data['Scores']


# In[13]:


sns.distplot(hours)


# In[14]:


sns.distplot(scores)


# # Linear Regression

# In[15]:


x=data.iloc[:, :-1].values
y=data.iloc[:,1].values


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=50)


# In[18]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)


# In[19]:


m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[20]:


y_pred=reg.predict(x_test)


# In[22]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[23]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# # What would be the predicted score if a student studies for 9.25 hours/day?

# In[24]:


h=9.25
s=reg.predict([[h]])
print("if a student studies for {} hours per day he/she will score {} % in exam".format(h,s))


# # Model Evaluation

# In[28]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 Score:',r2_score(y_test,y_pred))

