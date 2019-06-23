'''
XGBoost       :      0.13026 +/-      0.01377
LinXGBoost    :      0.11800 +/-      0.02570
Random Forests:      0.23034 +/-      0.01826
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
    param_grid = { "n_estimators": np.arange(30,55,3), # 48
                   "learning_rate": np.linspace(0.2,0.4,3), # 0.2
                   "min_child_weight": np.arange(2,6), # 5
                   "max_depth": np.arange(2,9,2), # 2
                   "subsample": np.linspace(0.6,0.8,3), # 0.6
                   "gamma": [ 0.03, 0.1, 0.3, 1 ] # 0.1
                  }
    grid_cv = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0., nthread=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, iid=True, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    reg = grid_cv.best_estimator_
    reg.fit(train_X, train_Y)
    xgb_pred_Y = reg.predict(test_X)

    # CV for LinXGBoost
    param_grid = { "learning_rate": [0.6, 0.7, 0.8], # 0.8
                   "gamma": [ 1, 3, 10 ], # 3 or 10
                   #"lbda": np.logspace(-3,-1,num=3), # -2
                   "min_samples_leaf": np.arange(40,61,5), #50
                   "n_estimators": [2,3]
                  }
    grid_cv = GridSearchCV(linxgb(max_depth=500, lbda=0.), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, iid=True, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    reg = grid_cv.best_estimator_
    reg.fit(train_X, train_Y)
    lin_pred_Y = reg.predict(test_X)

    # CV for Random Forest
    param_grid = { "n_estimators": np.arange(30,100,4), # 69 or 78
                   "min_samples_leaf": np.arange(1,3), # 1
                   "min_samples_split": np.arange(2,7), # 4 or 3
                   "max_depth": np.arange(10,31,2), # 24
                  }
    grid_cv = GridSearchCV(RandomForestRegressor(random_state=1), param_grid, scoring='neg_mean_squared_error', cv=cv_sets, iid=True, n_jobs=-1)
    grid_cv.fit(train_X, train_Y)
    reg = grid_cv.best_estimator_
    reg.fit(train_X, train_Y)
    rf_pred_Y = reg.predict(test_X)

    return nmse(test_Y,xgb_pred_Y), nmse(test_Y,lin_pred_Y), nmse(test_Y,rf_pred_Y)


if __name__ == '__main__':

    # read data file_name
    data = np.genfromtxt('fried_delve.data', delimiter = ' ')
    ndata, nfeatures = data.shape
    print("read {} samples, {} features".format(ndata,nfeatures))

    # predictions
    xgb_perf = []
    lin_perf = []
    rf_perf = []

    for k in range(0,20):
        print("starting {}-th iteration".format(k+1))

        np.random.seed(k)

        # Training & testing sets
        n_train_samples = 200
        p = np.random.permutation(ndata)
        p_train = p[0:n_train_samples]
        train_X = data[p_train,0:-1]
        train_Y = data[p_train,-1]
        p_test = p[n_train_samples:]
        test_X = data[p_test,0:-1]
        test_Y = 10*np.sin(np.pi*test_X[:,0]*test_X[:,1])+20*np.square(test_X[:,2]-0.5)+10*test_X[:,3]+5*test_X[:,4]

        # predictions
        xgb_nmse, lin_nmse, rf_nmse = compute(train_X,train_Y,test_X,test_Y)

        # bookkeeping
        xgb_perf.append(xgb_nmse)
        lin_perf.append(lin_nmse)
        rf_perf.append(rf_nmse)

        # print perf
        print("NMSE: XGBoost {:12.5f} LinXGBoost {:12.5f} Random Forests {:12.5f}". \
               format(xgb_nmse,lin_nmse,rf_nmse) )

    # Print stats
    print("XGBoost       : {:12.5f} +/- {:12.5f}".format(np.mean(xgb_perf),np.std(xgb_perf,ddof=1)) )
    print("LinXGBoost    : {:12.5f} +/- {:12.5f}".format(np.mean(lin_perf),np.std(lin_perf,ddof=1)) )
    print("Random Forests: {:12.5f} +/- {:12.5f}".format(np.mean(rf_perf),np.std(rf_perf,ddof=1)) )
