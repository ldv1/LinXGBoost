import numpy as np

class node:
    def __init__(self, verbose=0):
        self.parent = None
        self.left = None
        self.right = None
        self.split_value = 0.0
        self.split_feature = -1
        self.w = 0
        self.depth = 1
        self.gain = 0
        self.obj = 0
        self.verbose = verbose

    def is_leaf(self):
        return (self.left == None)

    def max_depth(self):
        if self.is_leaf():
            return self.depth
        else:
            return max( self.left.max_depth(), self.right.max_depth() )

    def num_leaves(self):
        if self.is_leaf():
            return 1
        else:
            return self.left.num_leaves()+self.right.num_leaves()

    def add(self):
        assert self.is_leaf(), "add children to a leaf!"

        left_node = node(verbose=self.verbose)
        left_node.parent_id = self
        left_node.depth = self.depth+1
        self.left = left_node

        right_node = node(verbose=self.verbose)
        right_node.parent_id = self
        right_node.depth = self.depth+1
        self.right = right_node

        return self.left, self.right

    def delete_node(self):
        assert self.is_leaf(), "delete a leaf!"
        assert self.depth != 1, "cannot delete root!"

        parent = self.parent
        parent.left = None
        parent.right = None

        return parent

    def objective(self, gamma=0.):
        if self.is_leaf():
            return self.obj+gamma
        else:
            return self.left.objective(gamma)+self.right.objective(gamma)

    def regularization(self, gamma=0., lbda=0.):
        if self.is_leaf():
            if np.isscalar(self.w):
                # in that case, lambda was set to 0 (no penalty for the bias)
                w2 = 0.
            else:
                w2 = np.dot(self.w,self.w)
            return gamma + 0.5*lbda*w2
        else:
            return self.left.regularization(gamma,lbda)+self.right.regularization(gamma,lbda)

    def predict(self,X):
        n = X.shape[0]
        if self.is_leaf():
            if np.isscalar(self.w):
                return self.w
            else:
                return np.dot(np.c_[X,np.ones(shape=(n,1), dtype=float)],self.w)
        else:
            assert self.split_feature > -1, "split feature must be > -1!"
            y = np.zeros(n, dtype=float)
            c = ( X[:,self.split_feature] < self.split_value )
            y[c]            = self.left.predict(X[c,:])
            y[np.invert(c)] = self.right.predict(X[np.invert(c),:])
            return y

    def find_best_split(self, X, g, h, lbda, gamma, max_samples_linear_model, min_samples_leaf):
        assert X.shape[0] == len(g)
        assert len(h) == len(g)
        assert self.split_feature == -1

        n, d = X.shape
        self.gain = float("-inf")
        for f in range(0,d): # split feature
            var = np.unique(X[:,f])
            mid_var = [ (x+y)/2. for x,y in zip(var[::],var[1::]) ]
            for pos in range(0,len(mid_var)): # split value
                c = ( X[:,f] < mid_var[pos] )
                left_n = np.sum(c)
                right_n = n-left_n
                if ( left_n < min_samples_leaf ) or ( right_n < min_samples_leaf ):
                    continue
                linear_model = ( left_n > d ) and ( left_n <= max_samples_linear_model )
                try:
                    _ , obj_left = self.get_weight( X[c,:], g[c], h[c], lbda, linear_model)
                except:
                    if self.verbose > 1:
                        print( "exception when testing for split of feature {} at pos {}".format(f,pos) )
                    continue
                c = np.invert(c)
                linear_model = ( right_n > d ) and ( right_n <= max_samples_linear_model )
                try:
                    _ , obj_right = self.get_weight( X[c,:], g[c], h[c], lbda, linear_model)
                except:
                    if self.verbose > 1:
                        print( "exception when testing for split of feature {} at pos {}".format(f,pos) )
                    continue
                gain = self.obj-(obj_left+obj_right+gamma)
                if gain > self.gain:
                    self.gain = gain
                    self.split_feature = f
                    self.split_value = mid_var[pos]

        if self.verbose > 1:
            if ( self.gain != float("-inf") ) and ( self.gain < 0. ):
                print( "negative gain!" )
            print( "find best split: gain={:+6.4e}, feature={:2d}, value={:+8.4f}".format(self.gain, self.split_feature, self.split_value) )

    def set_weight(self, X, g, h, lbda, linear_model):
        self.w, self.obj = self.get_weight(X, g, h, lbda, linear_model)

    def get_weight(self, X, g, h, lbda, linear_model):
        if linear_model:
            n,d = X.shape
            X_tilde = np.c_[X,np.ones(shape=(n,1), dtype=float)]
            g_tilde = np.dot(X_tilde.transpose(),g)
            H_tilde = np.dot(X_tilde.transpose()*h, X_tilde)
            #C = H_tilde+lbda*np.eye(d+1)
            Lambda = lbda*np.eye(d+1)
            Lambda[d,d] = 0.
            C = H_tilde+Lambda
            with np.errstate(divide='raise'):
                try:
                    cond = np.linalg.cond(C)
                except:
                    return self.get_weight(X, g, h, lbda, linear_model=False)
                    raise
            if cond > 1e12:
                return self.get_weight(X, g, h, lbda, linear_model=False)
            try:
                L = np.linalg.cholesky(C)
            except:
                if self.verbose > 1:
                    print( "C is not definite positive (X has {} instances)!".format(n) )
                return self.get_weight(X, g, h, lbda, linear_model=False)
                raise
            w = -np.linalg.solve(L.transpose(), np.linalg.solve(L,g_tilde))
            obj = 0.5*np.dot(g_tilde,w)
        else:
            g_tilde = np.sum(g)
            H_tilde = np.sum(h)
            #w = -g_tilde/(lbda+H_tilde)
            w = -g_tilde/H_tilde
            obj = 0.5*g_tilde*w

        return w, obj
