import sys
import numpy as np
import xgboost as xgb
from linxgb import linxgb
from metrics import *
from scipy.special import expit
import matplotlib.pyplot as plt

np.random.seed(0)

# Training set
train_n = 100
train_X = np.random.rand(train_n, 2)
c = ( np.square(train_X[:,0])+np.square(train_X[:,1]) < 0.5 )
train_Y = np.zeros(train_n)
train_Y[c] = 1
dtrain = xgb.DMatrix(train_X, label=train_Y.reshape(-1,1))

# Testing set
nx = ny = 100
X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
test_X = np.c_[ X.ravel(), Y.ravel() ]
dtest = xgb.DMatrix(test_X)

def plotsurf(ax, z):
    ax.axison = False
    Z = z.reshape(X.shape)
    red = [0.25,1,0.5]
    green = [1,0.3,0.3]
    ax.contourf(X, Y, Z, colors = [green,red], alpha = 0.2, levels = [0,0.5,1])
    ax.scatter(train_X[c,0],  train_X[c,1],  c = 'red', s = 20)
    ax.scatter(train_X[~c,0], train_X[~c,1], c = 'green',   s = 20)

# Common parameters, XGBoost 2 and LinXGBoost
num_trees = 1
learning_rate = 1
max_depth = 30
gamma = 5
subsample = 1.0
min_samples_leaf = 6

# XGBoost 1 (defaults with 50 trees) training
param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': 0.1, # step size shrinkage
         'objective': 'binary:logistic' # binary:logistic, reg:squarederror
         }
num_round = 50 # the number of round to do boosting, the number of trees
bst1 = xgb.train(param, dtrain, num_round)

# XGBoost 2 (single tree) training
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
         'objective': 'binary:logistic' # binary:logistic, reg:squarederror
         # 'eval_metric': the evaluation metric
         }
num_round = num_trees # the number of round to do boosting, the number of trees
bst2 = xgb.train(param, dtrain, num_round)

# LinXGBoost
linbst = linxgb(loss_function="binary:logistic",
                n_estimators=num_trees,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                max_samples_linear_model=10000,
                max_depth=max_depth,
                subsample=subsample,
                lbda=0,
                gamma=gamma,
                prune=True,
                verbose=1)
linbst.fit(train_X, train_Y)

# Plots
fig = plt.figure(figsize=(19,10), facecolor='white')
ax = fig.add_subplot(1, 3, 1, xticks=[], yticks=[])
z = bst1.predict(dtest)
z = np.array( [ row for row in z ] )
ax.set_title("XGBoost with 50 trees")
plotsurf(ax, z)
ax = fig.add_subplot(1, 3, 2, xticks=[], yticks=[])
z = bst2.predict(dtest)
z = np.array( [ row for row in z ] )
ax.set_title("XGBoost with 1 tree")
plotsurf(ax, z)
ax = fig.add_subplot(1, 3, 3, xticks=[], yticks=[])
z = linbst.predict(test_X)
ax.set_title("LinXGBoost with 1 tree")
plotsurf(ax, expit(z))

plt.show()
