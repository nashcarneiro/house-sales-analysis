# %%
# In this assignment, you are a Data Analyst working at a Real Estate Investment Trust.
# The Trust would like to start investing in Residential real estate. 
# You are tasked with determining the market price of a house given a set of features. 
# You will analyze and predict housing prices using attributes or features such as square footage, number of bedrooms, number of floors, and so on.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("kc_house_data_NaN.csv")

# %%
print(df.dtypes)
print(df.describe())
df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)
print(df.head())

# %%
df.replace('?', np.nan, inplace=True)
print(df.isnull().sum())

# %%
# Bedrooms and Bathrooms have missing values, since they are categorical values lets replace them with their mean
df['bathrooms'] = df['bathrooms'].replace(np.nan, df['bathrooms'].mean())
df['bedrooms'] = df['bedrooms'].replace(np.nan, df['bedrooms'].mean())
print(df.isnull().sum())

# %%
# Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a data frame. 
floors_df = df['floors'].value_counts().to_frame()
print(floors_df)

# %%
# Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers. 
sns.boxplot(x='waterfront', y='price', data=df)
plt.show()
# Houses without waterfront have lower prices but many outliers, whereas house with a water front are more expensice on average

# %%
# Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price. 
sns.regplot(x='sqft_above', y='price', data=df)
plt.ylim(0,)
plt.show()

# %%
df_numeric = df.select_dtypes(include=[np.number])
print(df_numeric.corr())

# %%
# Linear Regression Model
x = df[['sqft_living']]
y = df['price']
lm = LinearRegression()
lm.fit(x,y)
print("Score:",lm.score(x, y))

# %%
# Fit a linear regression model to predict the 'price' using the list of features:
features = ["floors","waterfront","lat","bedrooms","sqft_basement","view","bathrooms","sqft_living15","sqft_above","grade","sqft_living"]

# %%
z = df[features]
lm.fit(z, y)
print("score:",lm.score(z, y))

# %%
input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(input)
pipe.fit(z, y)
print("Score:",pipe.score(z, y))

# %%
x_data = df[features]
y_data = df['price']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("Number of training samples:", x_train.shape[0])
print("Number of test samples:", x_test.shape[0])

# %%
rm = Ridge(alpha=0.1)
rm.fit(x_train, y_train)
print("R^2 Score:",rm.score(x_test, y_test))

# %%
# Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object 
# using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. 

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
rm = Ridge(alpha=0.1)
rm.fit(x_train_pr, y_train)
print("Score on test data:", rm.score(x_test_pr, y_test))


