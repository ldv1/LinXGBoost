import numpy as np

def rmse(y,y_hat):
    return( np.sqrt(1.0/len(y)*np.sum( np.square(y.ravel()-y_hat.ravel()) ) ) )

def nmse(y,y_hat):
    y_bar = np.mean(y)
    return np.sum( np.square(y.ravel()-y_hat.ravel()) ) / np.sum( np.square(y.ravel()-y_bar) )
