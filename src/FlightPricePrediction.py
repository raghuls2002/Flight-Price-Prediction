#!/usr/bin/env python
# coding: utf-8

# ## 1. Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## 2. Loading the dataset.

# In[2]:


data = pd.read_csv('../data/dataset.csv')


# In[3]:


data.head()


# In[4]:


data = data.drop(['Unnamed: 0', 'flight'], axis=1)

data_original = data


# ## 3.1 Univariate Analysis

# ### Histogram

# In[5]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.hist(data['class'])
plt.title('Class distribution')

plt.subplot(1, 2, 2)
plt.hist(data['stops'])
plt.title('Stops distribution')

plt.show()


# ### Pie chart

# In[6]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.pie(data['airline'].value_counts(),autopct='%.2f', labels = data['airline'].unique())
plt.title('Airline distribution')

plt.subplot(1, 2, 2)
plt.pie(data['source_city'].value_counts(),autopct='%.2f', labels = data['source_city'].unique())
plt.title('Souce city distribution')

plt.show()


# ### Box plot

# In[7]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.boxplot(data['duration'])
plt.title('Duration distribution')

plt.subplot(1, 2, 2)
plt.boxplot(data['price'])
plt.title('Price distribution')

plt.show()


# ## 3.2 Bivariate Analysis

# ### Scatter plot

# In[8]:


sns.scatterplot(data, x='duration', y='price')


# ## 3.3 Multi - Variate Analysis

# ### Heatmap

# In[9]:


hm=data.corr()
sns.heatmap(hm)


# ## 4. Performing descriptive statistics on the dataset.

# In[10]:


data.describe()


# ## 5. Checking for Missing values.

# In[11]:


data.isnull().sum()


# ## 6. Checking for Categorical columns and performing encoding.

# In[12]:


# Checking data types of columns
column_types = data.dtypes

# Filtering categorical columns
categorical_columns = column_types[column_types == 'object'].index

# Printing categorical column names
print(categorical_columns)


# ### Label encoding

# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


le = LabelEncoder()


# In[15]:


for column in categorical_columns:
    data[column] = le.fit_transform(data[column])


# In[16]:


data.head()


# ## 7. Splitting the data into dependent and independent variables.

# In[17]:


y = data['price']

y.head()


# In[18]:


X= data.drop(columns=['price'],axis=1)

X.head()


# ## 8. Scaling the independent variables

# In[19]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_s = scaler.fit_transform(X)

X_s


# In[20]:


X = pd.DataFrame(X_s, columns = X.columns)

X.head()


# ## 9. Split the data into training and testing

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[23]:


X_train.head()


# In[24]:


X_test.head()


# In[25]:


y_train


# In[26]:


y_test


# ## 10. Building the Model

# In[27]:


from sklearn.ensemble import RandomForestRegressor


# In[28]:


model = RandomForestRegressor(n_estimators=100, random_state=42)


# ## 11. Training the Model

# In[29]:


model.fit(X_train, y_train)


# ## 12. Testing the Model

# In[30]:


y_pred = model.predict(X_test)

y_pred


# ## 14. Measure the performance using Metrics. 

# In[31]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[32]:


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[33]:


print("Mean Squared Error (MSE): ", mse)
print("Mean Absolute Error (MAE): ", mae)
print("R-squared (R2) Score: ", r2)


# ## 15. Saving the model

# In[34]:


import pickle


# In[35]:


with open('model.pickle', 'wb') as file:
   ### pickle.dump(model, file)

