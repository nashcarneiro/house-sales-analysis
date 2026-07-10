House Price Prediction Using Machine Learning

# Overview

This project analyzes residential housing data and develops machine learning models to predict house prices based on property characteristics such as living area, number of bedrooms and bathrooms, number of floors, waterfront access, location, condition, and overall grade.

The project simulates the role of a data analyst working for a Real Estate Investment Trust (REIT) that is considering investments in residential real estate. The objective is to explore the factors influencing house prices and build regression models capable of estimating property values.

# Project Objectives

The main objectives of this project are to:

* Clean and preprocess residential housing data.
* Identify and handle missing values.
* Explore relationships between housing features and price.
* Visualize price distributions and feature correlations.
* Build and evaluate multiple regression models.
* Compare simple linear regression with multivariable and polynomial regression.
* Apply Ridge regression to reduce overfitting through regularization.
* Evaluate model performance using the R² score on unseen test data.

# Dataset

The project uses the `kc_house_data_NaN.csv` dataset, which contains residential property data with features including:

* price — Sale price of the property.
* bedrooms — Number of bedrooms.
* bathrooms — Number of bathrooms.
* floors — Number of floors.
* waterfront — Whether the property has a waterfront view.
* view — Quality of the property's view.
* grade — Overall construction and design grade.
* sqft_living — Interior living space in square feet.
* sqft_above — Square footage above ground.
* sqft_basement — Basement square footage.
* sqft_living15 — Average living space of the 15 nearest neighboring properties.
* lat — Latitude of the property.

# Technologies and Libraries

The project was developed using Python and the following libraries:

* Pandas — Data manipulation and preprocessing.
* NumPy — Numerical operations and missing-value handling.
* Matplotlib — Data visualization.
* Seaborn — Statistical visualizations.
* Scikit-learn — Machine learning, preprocessing, pipelines, model training, and evaluation.

# Project Workflow

1. Data Loading and Initial Exploration

The housing dataset is loaded using Pandas. Data types, descriptive statistics, and the first rows of the dataset are examined to understand its structure.

Unnecessary columns such as `id` and `Unnamed: 0` are removed before further analysis.

2. Data Cleaning

Missing values are identified across the dataset. Missing bedroom and bathroom values are replaced using their respective column means to maintain the available observations.

3. Exploratory Data Analysis

Several exploratory analyses are performed to better understand the housing market data.

A box plot compares property prices for waterfront and non-waterfront homes, revealing differences in price distributions and outliers.

A regression plot is used to examine the relationship between above-ground square footage and house price, showing the direction and strength of the relationship.

A correlation matrix is also calculated for numerical features to identify variables that are strongly associated with house prices.

# Machine Learning Models

# Simple Linear Regression

A simple linear regression model is trained using `sqft_living` as the only predictor of house price.

This establishes a baseline model and demonstrates the relationship between living space and property value.

# Multiple Linear Regression

A second linear regression model uses multiple housing characteristics:

* Floors
* Waterfront access
* Latitude
* Bedrooms
* Basement area
* View quality
* Bathrooms
* Nearby living area
* Above-ground square footage
* Property grade
* Total living area

Using multiple features allows the model to capture more of the factors influencing house prices.

# Polynomial Regression Pipeline

A Scikit-learn pipeline combines three preprocessing and modeling steps:

1. Feature standardization using `StandardScaler`.
2. Polynomial feature generation using `PolynomialFeatures`.
3. Prediction using `LinearRegression`.

This allows the model to capture nonlinear relationships and interactions between housing features.

# Ridge Regression

The dataset is divided into training and testing sets, with 85% of observations used for training and 15% used for testing.

A Ridge regression model with a regularization parameter of `alpha=0.1` is trained on the training data and evaluated on previously unseen test data.

Ridge regression introduces regularization to control model complexity and help reduce overfitting.

# Polynomial Ridge Regression

The original features are transformed into second-degree polynomial features, allowing the model to capture nonlinear patterns and interactions between variables.

A Ridge regression model is then trained on the transformed training data and evaluated on the transformed test data using the R² score.

# Model Evaluation

Model performance is evaluated primarily using the **R² score**, which measures the proportion of variation in house prices explained by the model.

The project compares progressively more sophisticated approaches:

* Simple linear regression using one feature.
* Multiple linear regression using several housing characteristics.
* Polynomial regression.
* Ridge regression with regularization.
* Second-degree polynomial Ridge regression.

This progression demonstrates how feature selection, nonlinear transformations, and regularization can influence predictive performance.

# Key Insights

* Living area has a positive relationship with house price.
* Waterfront properties tend to have higher prices on average.
* Using multiple housing characteristics improves predictive capability compared with relying on a single feature.
* Polynomial transformations can capture nonlinear relationships and interactions between property characteristics.
* Ridge regularization helps control model complexity when working with a larger number of polynomial features.

