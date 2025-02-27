#!/usr/bin/env python
# coding: utf-8

# IMPORTING NECESSARY LIBRARIES.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from scipy.stats import gaussian_kde
import pickle


# READING CSV FILE. CONVERTING IT TO DATAFRAME AND PERFORMING EXPLORATORY ANALYSIS ON THE DATASET.

# In[2]:


house_price_dataset = pd.read_csv('Mumbai House Prices.csv')


# In[3]:


house_price_df = pd.DataFrame(house_price_dataset)


# In[4]:


house_price_df.shape


# In[5]:


house_price_df.head()


# In[6]:


house_price_df.isnull().sum()


# In[7]:


house_price_df.describe()


# CONVERTING CRORES TO LAKHS AND DROPPING THE price_unit COLUMN.

# In[8]:


house_price_df['price_in_Lakhs'] = house_price_df.apply(
    lambda row: row['price'] * 100 if row['price_unit'] == 'Cr' else row['price'],
    axis=1
)

house_price_df = house_price_df.drop(columns=['price', 'price_unit'])

house_price_df = house_price_df.rename(columns={'price_in_Lakhs': 'price'})

print(house_price_df.head())


# In[9]:


print(house_price_df.columns)


# DROPPING LOCALITY DUE TO HIGH CARDINALITY, AGE DUE TO REDUNDANCY AND STATUS DUE TO LIMITED CORRELATION WITH OTHER FEATURES.

# In[10]:


house_price_df = house_price_df.drop(columns=['age', 'locality', 'status'])


# In[11]:


house_price_df['price'] = np.log1p(house_price_df['price'])


# PLOTTING HISTOGRAM TO EXPLORE DISTRIBUTION OF PRICES.

# In[12]:


plt.hist(house_price_df['price'], bins=30, alpha=0.5, color='blue', edgecolor='black', density=True)

kde = gaussian_kde(house_price_df['price'])
x_vals = np.linspace(house_price_df['price'].min(), house_price_df['price'].max(), 100)
plt.plot(x_vals, kde(x_vals), color='red')  # KDE curve

plt.xlabel('Price (Lakhs)')
plt.ylabel('Density')
plt.title('Distribution of Price (Lakhs)')
plt.show()


# CREATING A NEW FEATURE, PRICE PER SQUARE FEET, WHICH IS USED FOR PROPERTY VALUATION IN PRACTICAL SETTINGS.

# In[13]:


house_price_df['price per square feet'] = house_price_df['price']/house_price_df['area']


# In[14]:


house_price_df.head()


# In[15]:


num_feats = house_price_df[['bhk', 'area', 'price', 'price per square feet']]
cat_feats = house_price_df.select_dtypes(include = ['object'])


# PLOTTING BOX PLOT TO CHECK FOR OUTLIERS.

# In[16]:


for i, column in enumerate(num_feats.columns):
    plt.figure(figsize = (10,6))
    plt.boxplot(house_price_df[column], vert = False)
    plt.title(f'Box plot of {column}')
    plt.xlabel('Value')
    plt.show()


# In[17]:


house_price_df_1 = house_price_df.drop(columns=['type', 'region'])
correlation = house_price_df_1.corr()
correlation


# HEATMAP TO DEMONSTRATE CORRELATION BETWEEN DIFFERENT FEATURES.

# In[18]:


plt.figure(figsize=(12, 12))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=False, cmap='Blues')
plt.title("Correlation Matrix of Encoded House Price Data")
plt.show()


# In[19]:


house_price_df = pd.concat([house_price_df_1, cat_feats], axis = 1)
house_price_df.head()


# In[20]:


house_price_df.region = house_price_df.region.apply(lambda x: x.strip())
loc_stats = house_price_df['region'].value_counts(ascending=False)
loc_stats


# In[21]:


loc_stats.values.sum()


# In[22]:


len(loc_stats[loc_stats>10])


# In[23]:


len(loc_stats[loc_stats<=10])


# In[24]:


location_stats_less_than_10 = loc_stats[loc_stats<=10]
location_stats_less_than_10


# In[25]:


len(house_price_df.region.unique())


# ASSIGNING ALL REGIONS WITH LESS THAN 10 PROPERTIES AS 'other' TO REDUCE THE SIZE OF THE DATAFRAME AND MAKE IT STATISTICALLY REFINED.

# In[26]:


house_price_df.region = house_price_df.region.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(house_price_df.region.unique())


# In[27]:


house_price_df.region.value_counts()


# In[28]:


target_col = 'price'
house_price_df = house_price_df[[col for col in house_price_df.columns if col != target_col] + [target_col]]
house_price_df.head()


# ONE-HOT ENCODING 'type' AND 'region' TO MAKE IT EASIER TO INCLUDE THOSE FEATURES IN MODEL TRAINING. 

# In[30]:


house_price_df_encoded = pd.get_dummies(house_price_df, columns=['type', 'region'], prefix=['type', 'region'])

house_price_df_encoded


# In[31]:


house_price_df_encoded.shape


# ASSIGNING INDEPENDENT FEATURES AND TARGET.

# In[32]:


X = house_price_df_encoded.drop('price', axis = 1)
Y = house_price_df_encoded['price']


# SPLITTING DATA IN TRAINING SET AND TEST SET.

# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[36]:


print(X.shape, X_train.shape, X_test.shape)


# DEFINING MODEL TO BE USED IN TRAINING.

# In[37]:


model = XGBRegressor()


# TRAINING THE MODEL.

# In[38]:


model.fit(X_train, Y_train)


# COMPARING Y_train WITH training_data_prediction AND Y_test WITH test_data_prediction.

# In[39]:


training_data_prediction = model.predict(X_train)


# In[40]:


print(training_data_prediction)


# In[41]:


score_1 = metrics.r2_score(Y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
print("R squared error: ", score_1)
print("Mean absolute error: ", score_2)


# In[42]:


test_data_prediction = model.predict(X_test)


# In[43]:


print(test_data_prediction)


# In[44]:


score_1_test = metrics.r2_score(Y_test, test_data_prediction)
score_2_test = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("R squared error: ", score_1_test)
print("Mean absolute error: ", score_2_test)


# In[45]:


df = pd.DataFrame({'Actual': np.round(Y_test, 2), 
                   'Predicted': np.round(test_data_prediction, 2)})
df.head(10)


# PLOTTING GRAPHS TO SHOW VARIOUS STATISTICAL INFERENCES AFTER TRAINING THE MODEL.

# In[46]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title("Actual vs Predicted Prices")
plt.show()


# In[47]:


plt.hist((test_data_prediction - Y_test), bins=50, color='purple', edgecolor='black', alpha=0.7)

plt.xlabel('Prediction Error (Predicted - Actual)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')

plt.show()


# In[48]:


with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'trained_model.pkl'")


# In[ ]:




