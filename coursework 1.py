# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:08:37 2023

@author: Julia
"""

#%% Reading data and importing all relevant libraries
from pandas import read_csv
import pandas as pd
from pandas import set_option
from matplotlib import pyplot
import numpy as np
filename = 'BITCOIN_dataset.csv' #accessing the file
data = read_csv(filename) #reading the file contents 
peek = data.head() #checking first 5 rows of data (from 0 to 4)
print(peek)
#%% Size of the data
shape = data.shape
print(shape) 
#%% Finding data types 
types = data.dtypes
print(types)
#%% Descriptive statistics
set_option('display.width', 100)
set_option('display.precision', 3)
description = data.describe()
print(description)
#%% Skew of univariate distributions
skew = data.skew()
print(skew)
#%% Histograms
data.hist(figsize = (13,13))
pyplot.show()

#%% Density plots
data.plot(kind = 'density',subplots = True, layout = (5,5), sharex = False, figsize = (18,18))
pyplot.show()

#%% BOX AND WHISKER PLOT
data.plot(kind = 'box',subplots = True, layout = (5,5),sharex = False, sharey = False,figsize = (10,10))
pyplot.show()
#%% Correlation and correlation matrix
data = data.drop('Date', axis=1) #removing the column 'Date' 
correlations = data.corr(method='pearson') #calculating correlations
#%%Visualising correlations
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0,22,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
pyplot.xticks(rotation=90)
pyplot.show()
pyplot.show()
#%%Finding the column number of Y
dataframe = read_csv(filename)
column_name = 'MKPRU'
column_index = dataframe.columns.get_loc(column_name)
print("The column of '{}' is: {}".format(column_name, column_index))
#%%
# 2. Separating array into input and output
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
print(dataframe.isna().sum()) #printing the sum number of missing values
dataframe = dataframe.apply(pd.to_numeric, errors='coerce') #converting the dataframe (type object) to numeric values
array = dataframe.values
#X is all the columns excluding the date and 10 column - Y
X = array[:, [i for i in range(1, 10)] + [j for j in range(11, array.shape[1])]]
Y = array[:,10]
#%%Removing missing values from X
# Selecing columns with missing values
df1 = pd.DataFrame(X)
print(df1.isna().sum()) #finding these columns in X
cols_with_missing = [7, 10, 11, 16]

# Creating a new DataFrame with missing values only in the selected columns
df_missing = pd.DataFrame(X[:, cols_with_missing], columns=cols_with_missing)
print (df_missing.dtypes) #checking data types before imputing values
df_imputed = df_missing.fillna(df_missing.mean()) #imputing the missing values with the mean of the column
print (df_imputed.dtypes) #checking data types after imputing

# Adding corrected columns to X
X[:, cols_with_missing] = df_imputed.values 
df2 = pd.DataFrame(X) #checking for missing values
print(df2.isna().sum())
#%%Normalising data before the feature selection
scaler = StandardScaler().fit(X) #enabling normal distribution
rescaledX = scaler.transform(X)

# 4. Summarising transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])
#%%Fearure selection
#Using SelectKBest because the Y variable is continuous
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions

# 3. Feature extraction
test = SelectKBest(score_func = f_classif, k = 6) #since the dataset has lots of variables we're selecting 6 major features
fit = test.fit(rescaledX,Y)

# 4. Summarise scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(rescaledX)

# 5. Summarise selected features
print(features[0:5,:])
#%%sFeature selection, trying to remove correlated features
# 3. Feature extraction
test = SelectKBest(score_func = f_classif, k = 4) #selecting 4 features now
fit = test.fit(rescaledX,Y)

# 4. Summarise scores
set_printoptions(precision=3)
print(fit.scores_)
features1 = fit.transform(rescaledX)

# 5. Summarise selected features
print(features1[0:5,:])
#%%
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
#%%
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, features1, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
