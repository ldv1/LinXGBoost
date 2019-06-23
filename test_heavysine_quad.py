import numpy as np
import xgboost as xgb
from linxgb import linxgb, make_polynomial_features
from metrics import *
from test_func import *
from test_plot import *

reg_func = "heavysine"

# Noise level
np.random.seed(0)
s2 = 0.05

# Training set
train_X, train_Y = test_func(reg_func, n_samples=201, var=s2)
train_X = make_polynomial_features(train_X,2)
dtrain = xgb.DMatrix(train_X, label=train_Y)

# Testing set
test_X, test_Y = test_func(reg_func, n_samples=5000)
test_X = make_polynomial_features(test_X,2)
dtest = xgb.DMatrix(test_X)

# Common parameters, XGBoost 2 and LinXGBoost
num_trees = 5
learning_rate = 1
max_depth = 30
gamma = 3
subsample = 1.0
min_samples_leaf = 6

# XGBoost 1 training
param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': 0.1, # step size shrinkage
         #'gamma': 1, # min. loss reduction to make another partition
         #'min_child_weight': min_samples_leaf, # In regression, minimum number of instances required in a child node
         #'max_depth': max_depth,  # maximum depth of a tree
         #'lambda': 0.0, # L2 regularization term on weights, default 0
         #'lambda_bias': 0.0, # L2 regularization term on bias, default 0
         #'save_period': 0, # 0 means do not save any model except the final round model
         #'nthread': 1,
         #'subsample': subsample,
         'objective': 'reg:squarederror' # binary:logistic, reg:squarederror
         # 'eval_metric': the evaluation metric
         }
num_round = 50 # the number of round to do boosting, the number of trees
bst1 = xgb.train(param, dtrain, num_round)

# XGBoost 2 training
param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': learning_rate, # step size shrinkage
         'gamma': gamma, # min. loss reduction to make another partition
         'min_child_weight': min_samples_leaf, # In regression, minimum number of instances required in a child node
         'max_depth': max_depth,  # maximum depth of a tree
         'lambda': 0.0, # L2 regularization term on weights, default 0
         'lambda_bias': 0.0, # L2 regularization term on bias, default 0
         'save_period': 0, # 0 means do not save any model except the final round model
         'nthread': 1,
         'subsample': subsample,
         'objective': 'reg:squarederror' # binary:logistic, reg:squarederror
         # 'eval_metric': the evaluation metric
         }
num_round = num_trees # the number of round to do boosting, the number of trees
bst2 = xgb.train(param, dtrain, num_round)

# LinXGBoost training
linbst = linxgb(n_estimators=num_trees,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                max_samples_linear_model=10000,
                max_depth=max_depth,
                subsample=subsample,
                lbda=0,
                gamma=gamma,
                verbose=1)
linbst.fit(train_X, train_Y)

# Make predictions
xgb1_pred_Y = bst1.predict(dtest)
xgb2_pred_Y = bst2.predict(dtest)
lin_pred_Y = linbst.predict(test_X)

# Print scores
print("NMSE: XGBoost1 {:12.5f}, XGBoost2 {:12.5f}, LinXGBoost {:12.5f}". \
       format(nmse(test_Y,xgb1_pred_Y),
              nmse(test_Y,xgb2_pred_Y),
              nmse(test_Y,lin_pred_Y)) )

# Plots
test_plot3a(reg_func, train_X[:,0], train_Y, test_X[:,0], test_Y,
            pred1_Y=xgb1_pred_Y, pred1_name="XGBoost",
            pred2_Y=xgb2_pred_Y, pred2_name="XGBoost",
            pred3_Y=lin_pred_Y,  pred3_name="LinXGBoost",
            fontsize=16, savefig=True)  # 36 for paper
