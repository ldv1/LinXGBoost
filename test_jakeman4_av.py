'''
11-x-11: XGBoost XX estimators, LinXGBoost 3 estimators
XGBoost       :      0.57775 +/-      0.02140
LinXGBoost    :      0.61580 +/-      0.02625
Random Forests:      0.59069 +/-      0.01945

41-x-41: XGBoost 88 estimators, LinXGBoost 3 estimators
XGBoost       :      0.13347 +/-      0.00211
LinXGBoost    :      0.13605 +/-      0.00257
Random Forests:      0.13759 +/-      0.00222
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
    param_grid = { "n_estimators": np.arange(20,52,3), # 28, 35
                   "learning_rate": np.linspace(0.3,0.6,4), # 0.2, 0.2
                   "min_child_weight": np.arange(1,6), # 2,4
                   "max_depth": np.arange(2,9,2), # 8,4
                   "subsample": np.linspace(0.7,1,4), # 1, 0.9
                   "gamma": [ 0.1, 0.3, 1 ] # 0.1, 0.3
                  }
    grid_cv = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0., nthread=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, iid=True, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    reg = grid_cv.best_estimator_
    reg.fit(train_X, train_Y)
    xgb_pred_Y = reg.predict(test_X)

    # CV for LinXGBoost
    param_grid = { "n_estimators": [2,3],
                   "learning_rate": [0.8,0.9],
                   "gamma": [ 0.1, 0.3, 1, 3 ],
                   "lbda": np.logspace(-7,-1,num=4),
                   "min_samples_leaf": [3,4,8,16],
                  }
    grid_cv = GridSearchCV(linxgb(max_depth=200), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, iid=True, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    reg = grid_cv.best_estimator_
    reg.fit(train_X, train_Y)
    lin_pred_Y = reg.predict(test_X)

    # CV for Random Forest
    param_grid = { "n_estimators": np.arange(10,40,4), # 53, 44
                   "min_samples_leaf": np.arange(1,4), # 1, 1
                   "min_samples_split": np.arange(2,5), # 2, 5
                   "max_depth": np.arange(2,13,2), # 16, 6
                  }
    grid_cv = GridSearchCV(RandomForestRegressor(random_state=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, iid=True, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    reg = grid_cv.best_estimator_
    reg.fit(train_X, train_Y)
    rf_pred_Y = reg.predict(test_X)

    return nmse(test_Y,xgb_pred_Y), nmse(test_Y,lin_pred_Y), nmse(test_Y,rf_pred_Y)


if __name__ == '__main__':

    reg_func = "jakeman4"

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
        n_train_samples = 11
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
