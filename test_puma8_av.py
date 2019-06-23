'''
XGBoost       :      0.03741 +/-      0.00144
LinXGBoost    :      0.04014 +/-      0.00142
Random Forests:      0.03898 +/-      0.00138
'''

import time
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from linxgb import linxgb, make_polynomial_features
from metrics import *
from test_func import *
from test_plot import *

def compute(train_X,train_Y,test_X,test_Y):
    # CV parameters
    cv_sets = KFold(n_splits=3, shuffle=True, random_state=1)

    # least-square fit
    reg = linear_model.LinearRegression()
    reg.fit(train_X,train_Y)
    lsf_pred_Y = reg.predict(test_X)

    # CV for XGBoost
    param_grid = { "n_estimators": np.arange(130,166,5), # 48
                   "learning_rate": [0.05,0.10,0.15], # 0.2
                   "min_child_weight": np.arange(3,6), # 5
                   "max_depth": np.arange(4,11,2), # 2
                   "subsample": np.linspace(0.6,0.8,3), # 0.6
                   "gamma": [ 0.001, 0.003, 0.01 ] # 0.1
                  }
    grid_cv = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0., nthread=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    #reg = grid_cv.best_estimator_
    #reg.fit(train_X, train_Y)
    xgb_pred_Y = grid_cv.predict(test_X)

    # CV for LinXGBoost
    param_grid = { "learning_rate": [0.4,0.5], # 0.8
                   "gamma": [ 1, 30, 100 ], # 3 or 10
                   #"lbda": np.logspace(-13,-2,num=3), # -2
                   "min_samples_leaf": [16,24,32], #50
                  }
    grid_cv = GridSearchCV(linxgb(n_estimators=3,max_depth=200,lbda=0.), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    #reg = grid_cv.best_estimator_
    #reg.fit(train_X, train_Y)
    lin_pred_Y = grid_cv.predict(test_X)

    # CV for Random Forest
    param_grid = { "n_estimators": np.arange(80,121,5), # 69 or 78
                   "min_samples_leaf": np.arange(1,4), # 1
                   "min_samples_split": np.arange(2,5), # 4 or 3
                   "max_depth": np.arange(12,27,2), # 24
                  }
    grid_cv = GridSearchCV(RandomForestRegressor(random_state=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    #reg = grid_cv.best_estimator_
    #reg.fit(train_X, train_Y)
    rf_pred_Y = grid_cv.predict(test_X)

    return nmse(test_Y,lsf_pred_Y), nmse(test_Y,xgb_pred_Y), nmse(test_Y,lin_pred_Y), nmse(test_Y,rf_pred_Y)


if __name__ == '__main__':

    # read data file_name
    data = np.loadtxt("pumadyn-8nm.data")
    features = data[:,:-1]
    features -= np.mean(features,axis=0)
    target = data[:,-1]

    # predictions

    lsf_perf = []
    xgb_perf = []
    lin_perf = []
    rf_perf = []

    for k in range(0,20):
        print( "starting {}-th iteration".format(k+1) )

        np.random.seed(k)

        # Training & testing sets
        train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.3, random_state=k)

        # predictions
        lsf_nmse, xgb_nmse, lin_nmse, rf_nmse = compute(train_X,train_Y,test_X,test_Y)

        # bookkeeping
        lsf_perf.append(lsf_nmse)
        xgb_perf.append(xgb_nmse)
        lin_perf.append(lin_nmse)
        rf_perf.append(rf_nmse)

        # print perf
        print("NMSE: LSF {:12.5f} XGBoost {:12.5f} LinXGBoost {:12.5f} Random Forests {:12.5f}". \
               format(lsf_nmse,xgb_nmse,lin_nmse,rf_nmse) )

    # Print stats
    print("LSF           : {:12.5f} +/- {:12.5f}".format(np.mean(lsf_perf),np.std(lsf_perf,ddof=1)) )
    print("XGBoost       : {:12.5f} +/- {:12.5f}".format(np.mean(xgb_perf),np.std(xgb_perf,ddof=1)) )
    print("LinXGBoost    : {:12.5f} +/- {:12.5f}".format(np.mean(lin_perf),np.std(lin_perf,ddof=1)) )
    print("Random Forests: {:12.5f} +/- {:12.5f}".format(np.mean(rf_perf),np.std(rf_perf,ddof=1)) )
