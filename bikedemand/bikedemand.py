import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score



df_train = pd.read_csv('train.csv', header = 0)

df_test = pd.read_csv('test.csv', header = 0)

df_train['month'] = pd.DatetimeIndex(df_train.datetime).month
df_train['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
df_train['hour'] = pd.DatetimeIndex(df_train.datetime).hour

df_train = df_train.drop(['datetime','casual','registered'], axis = 1)

df_train_target = df_train['count'].values
df_train_data = df_train.drop(['count'],axis = 1).values
print 'df_train_data shape is ', df_train_data.shape
print 'df_train_target shape is ', df_train_target.shape
#==============================================================================

df_test['month'] = pd.DatetimeIndex(df_test.datetime).month
df_test['day'] = pd.DatetimeIndex(df_test.datetime).dayofweek
df_test['hour'] = pd.DatetimeIndex(df_test.datetime).hour

df_test_data = df_test.drop(['datetime'], axis = 1).values

print 'df_test_data shape is ', df_test_data.shape
#===============================================================================

cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=1, test_size=0.2,
    random_state=0)

print "SVR(kernel='rbf',C=10,gamma=.001)"
'''for train, test in cv:
    
    svc = svm.SVR(kernel ='rbf', C = 10, gamma = .001).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

print "Ridge"    
for train, test in cv:    
    svc = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))
'''
print "Random Forest(n_estimators = 100)"    
for train, test in cv:    
    svc = RandomForestRegressor(n_estimators = 100).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

soln = svc.predict(df_test_data)
print soln.shape

my_solution = pd.DataFrame(soln,columns = ["count"])
dg_test1 = pd.DataFrame(df_test)
dg_test1['count'] = my_solution['count']
dg_test1.drop(dg_test1.columns[1:12], axis =1)
ans = dg_test1[['datetime', 'count']]

ans.to_csv("solution_one.csv")
