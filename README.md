# LinXGBoost
Extension of the awesome XGBoost to linear models at the leaves

## Motivation
XGBoost is often presented as the algorithm that wins
every ML competition. Surprisingly, this is true even though predictions are piecewise constant.
This might be justified in high dimensional
input spaces, but when the number of features is low, a piecewise linear model is likely to perform better. XGBoost was extended into LinXGBoost that stores
at each leaf a linear model. This extension is equivalent to piecewise
regularized least-squares.

## Dependencies
You will need python 3 with
[numpy](http://www.numpy.org/).
If you intend to run the tests, then
[sklearn](http://scikit-learn.org/stable/),
[matplotlib](https://matplotlib.org/), [XGBoost](https://github.com/dmlc/xgboost)
and [pandas](http://pandas.pydata.org/) must be installed.

## Code
The class `linxgb` is, without surprise, defined in `linxgb.py`.
It is the counterpart of `xgb.XGBRegressor`: XGBoost for regression using a sklearn-like API (see [Scikit-Learn API](http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)). As such, it implements two methods: `fit`
and `predict`. Consequently, you can use sklearn for cross-validation.

The definition of a tree is in `node.py`. Normally, you should not have to instance a tree directly.

## Code example
Suppose `train_X` (a numpy array) and `train_Y` (a numpy vector) are the training data sets: Inputs and labels, respectively. Then the following will fit a LinXGBoost model with 3 estimators (or trees):

```python
reg = linxgb(n_estimators=3)
reg.fit(train_X, train_Y)
```

For the predictions, it is as simple as:

```python
pred_Y = reg.predict(test_X)
```

## Parameters
Most significant parameters comply with
[XGBoost parameter](http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) definition. They are:
* n_estimators [default=5]
  - Number of trees to fit.
  - range: [1,+∞]
* learning_rate [default=0.3]
  - step size shrinkage used in update to prevents overfitting.
  - range: [0,1]
* gamma [default=0]
  - penalty for the number of leaves of a tree.
  - range: [0,+∞]
* max_depth [default=6]
  - maximum depth of a tree (opposite to XGBoost, 0 does not indicate no limit).
  - range: [0,+∞]
* subsample [default=1]
  - subsample ratio of the training instance.
  - range: (0,1]
* lbda [default=1]
  - L2 regularization term on weights.
  - range: [0,+∞]
* nthread [default=1]
  - Number of parallel threads used to run LinXGBoost; LinXGBoost is currently not multi-threaded and so this parameter has no effect.
  - range: [1,+∞]
* random_state [default=None]
  - random number seed.
  - range: [0,+∞]


Additionally, we have:
* loss_function [default="reg:squarederror"]
  - in XGBoost, it is objective: The objective to minimize.
  - It is either "reg:squarederror" for regression or "binary:logistic"
  for binary classification.
* min_samples_split [default=3]
  - the minimum number of samples required to split an internal node.
  - range: [2,+∞]
* min_samples_leaf [default=2]
  - the minimum number of samples required to be at a leaf node.
  - range: [1,+∞]
* min_split_loss [default=0]
  - the minimum loss for a split; it is currently without effect.
  - range: [-∞,+∞]
* max_samples_linear_model [default=the biggest int]
  - if the number of input data points in a node is above this threshold, then the split is done according to the XGBoost strategy: constant values at the nodes.
  - range: (0,∞]
* prune [default=True]
  - if true, then subtrees that do not lead to a decrease of the objective function are pruned out.
  - range: [True,False]
* verbose [default=0]
  - the verbosity: 0 is for the least verbose mode.
  - range: [0,+∞]

## Tests
Several tests can be run:
* `test_heavysine.py`: A one-dimensional problem (see [Adapting to Unknown Smoothness via Wavelet Shrinkage](http://statweb.stanford.edu/~imj/WEBLIST/1995/ausws.pdf) by Donoho and Johnstone)
* `test_jakeman1_av.py`: The f1 response surface from [Local and Dimension Adaptive Sparse Grid Interpolation and Quadrature](https://arxiv.org/pdf/1110.0010.pdf).
* `test_jakeman4_av.py`: The f4 response surface from [Local and Dimension Adaptive Sparse Grid Interpolation and Quadrature](https://arxiv.org/pdf/1110.0010.pdf) with w1=0.5 and w2=3.
* `test_friedman1_av.py`: The Friedman 1 data set is a synthetic dataset. It has been previously employed in evaluations of MARS (Multivariate Adaptive Regression Splines by Friedman) and bagging (Breiman, 1996). It is particularly suited to examine the ability of methods to uncover interaction effects that are present in the data.
* `test_ccpp_av.py`: This real-world dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant.
* `test_puma8_av.py`: This problem was generated using a robot-arm simulation. The data set is highly non-linear and has very low noise. It contains 8192 data samples with 8 attributes.

## Authors
Laurent de Vito

## License
All third-party libraries are subject to their own license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
