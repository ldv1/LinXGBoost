'''
11-x-11: XGBoost 54 estimators, LinXGBoost 3 estimators
XGBoost       :      0.08579 +/-      0.00260
LinXGBoost    :      0.07929 +/-      0.00498
Random Forests:      0.08381 +/-      0.00198

41-x-41: XGBoost 88 estimators, LinXGBoost 5 estimators
XGBoost       :      0.00829 +/-      0.00022
LinXGBoost    :      0.00576 +/-      0.00050
Random Forests:      0.01133 +/-      0.00025
'''

import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xgboost as xgb
from linxgb import linxgb, make_polynomial_features
from metrics import *
from test_func import *
from test_plot import *

def compute(train_X,train_Y,test_X,test_Y):
    # CV parameters
    cv_sets = KFold(n_splits=10, shuffle=True, random_state=1)

    # CV for XGBoost
    param_grid = { "n_estimators": np.arange(30,59,2), # 54, 88
                   "learning_rate": np.linspace(0.2,0.4,3), # 0.3, 0.1
                   "min_child_weight": np.arange(2,5), # 3,3
                   "max_depth": np.arange(4,13,2), # 8,12
                   "subsample": np.linspace(0.6,0.8,3), # 0.7, 0.7
                   "gamma": [ 0.01, 0.03, 0.1, 0.3 ] # 0.1, 0.3
                  }
    grid_cv = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0., nthread=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    xgb_pred_Y = grid_cv.predict(test_X)

    # CV for LinXGBoost
    param_grid = { "learning_rate": [0.4, 0.5, 0.6], # 0.7, 0.6
                   "gamma": [ 0.3, 1, 3 ], # 0.3, 0.6
                   #"lbda": np.logspace(-4,-2,num=3), # -2, -3
                   "min_samples_leaf": [4,6,8], # 2, 2
                  }
    grid_cv = GridSearchCV(linxgb(max_depth=200,n_estimators=5,lbda=0.), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    lin_pred_Y = grid_cv.predict(test_X)

    # CV for Random Forest
    param_grid = { "n_estimators": np.arange(20,60,3), # 22, 65
                   "min_samples_leaf": np.arange(1,3), # 1, 1
                   "min_samples_split": np.arange(2,4), # 2, 2
                   "max_depth": np.arange(10,40,3), # 14, 26
                  }
    grid_cv = GridSearchCV(RandomForestRegressor(random_state=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    rf_pred_Y = grid_cv.predict(test_X)

    return nmse(test_Y,xgb_pred_Y), nmse(test_Y,lin_pred_Y), nmse(test_Y,rf_pred_Y)


if __name__ == '__main__':

    reg_func = "jakeman1"

    # Noise level
    np.random.seed(0)
    s2 = 0.05

    # Testing set on the square
    n_test_samples = 1001
    test_X, test_Y = test_func(reg_func, n_samples=n_test_samples)

    # predictions

    xgb_perf = []
    lin_perf = []
    rf_perf = []

    for k in range(0,20):
        print("starting {}-th iteration".format(k+1))

        np.random.seed(k)

        # Training set on the square
        n_train_samples = 41
        train_X, train_Y = test_func(reg_func, n_samples=n_train_samples, var=s2)

        # predictions
        xgb_nmse, lin_nmse, rf_nmse = compute(train_X,train_Y,test_X,test_Y)

        # bookkeeping
        xgb_perf.append(xgb_nmse)
        lin_perf.append(lin_nmse)
        rf_perf.append(rf_nmse)

        # print perf
        print("NMSE: XGBoost {:12.5f} LinXGBoost {:12.5f} Random Forests {:12.5f}". \
               format(xgb_nmse,lin_nmse,rf_nmse))

    # Print stats
    print("XGBoost       : {:12.5f} +/- {:12.5f}".format(np.mean(xgb_perf),np.std(xgb_perf,ddof=1)))
    print("LinXGBoost    : {:12.5f} +/- {:12.5f}".format(np.mean(lin_perf),np.std(lin_perf,ddof=1)))
    print("Random Forests: {:12.5f} +/- {:12.5f}".format(np.mean(rf_perf),np.std(rf_perf,ddof=1)))
