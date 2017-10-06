import numpy as np

# Parameters of the sine function
fr = 3
damp_fr  = 1.5
damp_amp = 0.8

# Test functions
def test_func(name, n_samples, var=0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    assert name in ["lo1", "linear", "sine", "heavysine","friedman1","square","jakeman1","jakeman4","jakeman4s","jakeman41","jakeman41s","island"], "unknown test function!"
    if name in ["lo1", "linear", "sine", "heavysine"]:
        X = np.linspace(0,1,n_samples)
        if name == "linear":
            y = 2.1*X-1.4
        elif name == "sine":
            y = np.exp(damp_amp*X)*np.sin(2*np.pi*np.exp(damp_fr*X)*X)
        elif name == "heavysine":
            y = 4*np.sin(4*np.pi*X)-np.sign(X-0.3)-np.sign(0.72-X)
        elif name == "lo1":
            y = 1./(np.abs(0.3-np.square(X))+0.1)
        X = X.reshape(-1,1)
    elif name == "square":
        X1,X2 = np.meshgrid(np.linspace(0,1,n_samples),np.linspace(0,1,n_samples))
        Y = np.zeros(X1.shape)
        Y[(X1>0.5) & (X2>0.5)] = 10.
        X = np.c_[X1.ravel(),X2.ravel()]
        y = Y.ravel().reshape(-1,1)
    elif name == "jakeman1":
        X1,X2 = np.meshgrid(np.linspace(0,1,n_samples),np.linspace(0,1,n_samples))
        Y = 1./(np.abs(0.3-np.square(X1)-np.square(X2))+0.1)
        X = np.c_[X1.ravel(),X2.ravel()]
        y = Y.ravel()
    elif name == "jakeman4":
        X1,X2 = np.meshgrid(np.linspace(0,1,n_samples),np.linspace(0,1,n_samples))
        Y = np.exp(0.5*X1+3*X2)
        Y[X1>0.5] = 0.
        Y[X2>0.5] = 0.
        X = np.c_[X1.ravel(),X2.ravel()]
        y = Y.ravel()
    elif name == "jakeman4s": # random sampling in the unit square
        X1 = np.random.rand(n_samples*n_samples)
        X2 = np.random.rand(n_samples*n_samples)
        X = np.c_[X1,X2]
        y = np.exp(0.5*X1+3*X2)
        y[X1>0.5] = 0.
        y[X2>0.5] = 0.
    elif name == "jakeman41":
        X1,X2 = np.meshgrid(np.linspace(0,1,n_samples),np.linspace(0,1,n_samples))
        Y = 10*np.sin(3*np.pi*X1+np.pi*X2)
        C = (X1>0.6) & (X2<0.5)
        Y[C] = -2*X1[C]+X2[C]
        C = (X1<0.4) & (X2>0.5)
        Y[C] = 3*X1[C]-2*X2[C]
        C = (X1>0.4) & (X2>0.5)
        Y[C] = np.exp(X1[C]+2*X2[C])
        X = np.c_[X1.ravel(),X2.ravel()]
        y = Y.ravel()
    elif name == "jakeman41s":
        X1 = np.random.rand(n_samples,n_samples)
        X2 = np.random.rand(n_samples,n_samples)
        Y = 10*np.sin(3*np.pi*X1+np.pi*X2)
        C = (X1>0.6) & (X2<0.5)
        Y[C] = -2*X1[C]+X2[C]
        C = (X1<0.4) & (X2>0.5)
        Y[C] = 3*X1[C]-2*X2[C]
        C = (X1>0.4) & (X2>0.5)
        Y[C] = np.exp(X1[C]+2*X2[C])
        X = np.c_[X1.ravel(),X2.ravel()]
        y = Y.ravel()
    elif name == "island":
        X1,X2 = np.meshgrid(np.linspace(-1,1,n_samples),np.linspace(-1,1,n_samples))
        Y = np.ones(X1.shape)
        C = X1*X1+X2*X2
        Y[C>0.5**2] = 0.
        X = np.c_[X1.ravel(),X2.ravel()]
        y = Y.ravel()
    elif name == "friedman1":
        X = np.random.rand(n_samples,10)
        y = 10*np.sin(np.pi*X[:,0]*X[:,1])+20*np.square(X[:,2]-0.5)+10*X[:,3]+5*X[:,4]

    y += np.sqrt(var)*np.random.randn(*y.shape)
    return X,y

def test_func_lineX(name, n_samples):
    assert name in ["square", "jakeman1", "jakeman4", "jakeman4s", "jakeman41", "jakeman41s","island"], "unknown test function!"
    if name == "square":
        X = np.linspace(0,1,n_samples)
        y = np.zeros(X.shape)
        y[X>0.5] = 10.
    elif name == "jakeman1":
        X1 = X2 = np.linspace(0,1,n_samples)
        y = 1./(np.abs(0.3-np.square(X1)-np.square(X2))+0.1)
        X = np.c_[X1.ravel(),X2.ravel()]
    elif name == "jakeman4" or name == "jakeman4s":
        X2 = np.linspace(0,1,n_samples)
        X1 = 0.1*np.ones(n_samples)
        y = np.exp(0.5*X1+3*X2)
        y[X2>0.5] = 0.
        X = np.c_[X1.ravel(),X2.ravel()]
    elif name == "jakeman41" or name == "jakeman41s":
        X2 = np.linspace(0,1,n_samples)
        X1 = 0.2*np.ones(n_samples)
        y = 10*np.sin(3*np.pi*X1+np.pi*X2)
        C = (X1>0.6) & (X2<0.5)
        y[C] = -2*X1[C]+X2[C]
        C = (X1<0.4) & (X2>0.5)
        y[C] = 3*X1[C]-2*X2[C]
        C = (X1>0.4) & (X2>0.5)
        y[C] = np.exp(X1[C]+2*X2[C])
        X = np.c_[X1.ravel(),X2.ravel()]
    elif name == "island":
        X1 = X2 = np.linspace(-1,1,n_samples)
        y = np.zeros(X1.shape)
        C = X1*X1+X2*X2
        y[C>0.5**2] = 0.
        X = np.c_[X1.ravel(),X2.ravel()]

    return X,y
