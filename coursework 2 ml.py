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
import time
start_time = time.time()
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
#Applying models for prediction
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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
#Using train-test split to enhance the speed from this method
#To decrease variance from k-fold
X = rescaledX #changing the variable X = features, equal to 6 features
#changing the variable X = features1, equal to 4 features
from sklearn.model_selection import train_test_split
validation_size = 0.2 #test data (validation dataset)
seed = 7
X_train,X_validation,Y_train, Y_validation = train_test_split(X,Y,test_size = validation_size,random_state = seed)
#%%
#Assesing models performance with mean squared error
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    #%%
#Assesing models performance with R2
results = []
names = []
scoring = 'r2'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name) 
    r2 = "%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print('R2 results for the model:', r2)
#%%
#Alternative way of printing the scores
#from sklearn.model_selection import cross_validate
#scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}

#for name, model in models:
    #results = cross_validate(model,  X_train, Y_train, cv=kfold, scoring=scoring)
    #print('Model:', name)
    #print('Mean negative mean squared error:', np.mean(results['test_neg_mean_squared_error']))
    #print ('Std:', np.std(results['test_neg_mean_squared_error']))
    #print('Mean R^2 score:', np.mean(results['test_r2']))
    #print()
    #%%
#Plotting the results with R2 score
fig = pyplot.figure(figsize = (10,10))
fig.suptitle('Scale algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
pyplot.ylim(0.94, 1.0) #Setting the y limits to better show the R2 of models
ax.set_xticklabels(names)
pyplot.show()
#%%
#Testing the model performance with the pipeline on the features
#In the pipelines standardisation is applied
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from numpy import set_printoptions

pipelines = []
pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()),('FeatureSelection', SelectKBest(score_func=f_regression, k=6)),('LR',LinearRegression())])))
#after the tuning, k=8
pipelines.append(('ScaledLASSO',Pipeline([('Scaler',StandardScaler()),('FeatureSelection', SelectKBest(score_func=f_regression, k=6)),('LASSO',Lasso())])))
pipelines.append(('ScaledEN',Pipeline([('Scaler',StandardScaler()),('FeatureSelection', SelectKBest(score_func=f_regression, k=6)),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()),('FeatureSelection', SelectKBest(score_func=f_regression, k=6)),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART',Pipeline([('Scaler',StandardScaler()),('FeatureSelection', SelectKBest(score_func=f_regression, k=6)),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR',Pipeline([('Scaler',StandardScaler()),('FeatureSelection', SelectKBest(score_func=f_regression, k=6)),('SVR',SVR())])))

# Splitting the data into train and test sets
X = rescaledX
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# Performing k-fold cross-validation on the pipeline
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = []
names = []
for name, pipeline in pipelines:
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    print('%s: Mean %f (Std %f)' % (name, cv_results.mean(), cv_results.std()))
#%%
#Assesing the pipelines with R2
results1 = []
names1 = []
scoring = 'r2'
for name, pipeline in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring=scoring)
    results1.append(cv_results)
    names1.append(name) 
    r2 = "%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print('R2 results for the model:', r2)
#%%
#Plotting the R2 results
fig = pyplot.figure(figsize = (10,10))
fig.suptitle('Scale algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results1)
pyplot.ylim(0.98, 1.0)
ax.set_xticklabels(names1)
pyplot.show()
#%%
#Tuning  Linear Regression
from sklearn.model_selection import GridSearchCV

# Defining the grid of hyperparameters of feature selection
param_grid = {'FeatureSelection__k': [4, 6, 8, 14]}

# Defining the GridSearchCV object from the pipelines and scoring
grid = GridSearchCV(estimator=pipelines[0][1], param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)

# Fitting the GridSearchCV object on the training data
grid_result = grid.fit(X_train, y_train)

# Printing the best score and hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means,stds,params):
    print("%f (%f) with: %r" %(mean,stdev,param))
#%%
#Applying Lasso feature selection to deal with correlation of features
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# Create a LassoCV model
lasso = LassoCV(cv=5)

# Fit the model to the training data
lasso.fit(rescaledX, Y)
df_scaled = pd.DataFrame(rescaledX)

# Get the selected features
selected_features = pd.DataFrame({'Feature': df_scaled.columns, 'Coefficient': lasso.coef_})
selected_features = selected_features.loc[selected_features['Coefficient'] != 0, 'Feature'].tolist()
print(selected_features)
#%% Applying ensemble methods
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
ensembles = []
ensembles.append(('ScaledAB',Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('ScaledGBM',Pipeline([('Scaler',StandardScaler()),('GBM',GradientBoostingRegressor())])))
ensembles.append(('ScaledRF',Pipeline([('Scaler',StandardScaler()),('RF',RandomForestRegressor())])))
ensembles.append(('ScaledET',Pipeline([('Scaler',StandardScaler()),('ET',ExtraTreesRegressor())])))

results = []
names = []
for name,model in ensembles:
    kfold = KFold(n_splits=10, random_state = 7, shuffle=True)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring='neg_mean_squared_error') #changing to 'r2' for r2 results scoring='r2'
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)
#%%#%%73 minutes part = advises not to run
#Tuning Extra Trees Regressor model for better performance 
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400], 
              'max_depth': [None, 5, 10, 20, 30, 40],
              'min_samples_split': [2, 5, 10, 20]}
model = ExtraTreesRegressor(random_state=seed)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#%%
#Running the final tuned model
#Defining Extra Trees Regressor with best parameters
from sklearn.metrics import mean_squared_error, r2_score

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = ExtraTreesRegressor(n_estimators=300, max_depth=20, min_samples_split=2, random_state=seed)
model.fit(rescaledX,y_train)


# Transform the validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
mse = mean_squared_error(y_test, predictions)
std = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean squared error: %.3f" % mse)
print("Standard deviation: %.3f" % std)
print("R2 score: %.3f" % r2)
#%%Visualising the predicted vs. actual values
import matplotlib.pyplot as plt
#Sorting values in ascending order to visualise 
sort_indices = y_test.argsort() 
y_test_sorted = y_test[sort_indices]
predictions_sorted = predictions[sort_indices]

# Plotting the actual Y values as a line graph
plt.plot(y_test_sorted, label='Actual Y', color='blue')

# Plotting the predicted Y values on the same graph as a line graph
plt.plot(predictions_sorted, label='Predicted Y', color='orange')

plt.xlabel('Data Point Index')
plt.ylabel('Y Value')
plt.title('Actual Y vs. Predicted Y')
plt.legend()
plt.show()
#%%Plotting the predicted vs. actual values with a linear regression
plt.scatter(y_test, predictions, label='Data Points')
plt.plot(y_test, y_test, color='red', label='Linear Regression Line')
#setting actual Y as linear regression line
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Actual Y vs. Predicted Y')
plt.legend()
plt.show()
#%%Optimisation 
#Measuring the time that it takes the code to run
end_time = time.time()
time_taken = (end_time - start_time) / 60
print("Time taken:", time_taken, "minutes")