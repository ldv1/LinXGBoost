import sys
import numpy as np
from node import node
from sklearn.preprocessing import PolynomialFeatures

def make_polynomial_features(X,order):
    """Add polynomial features to a design matrix.

    Users must explicitly add polynomial features if they wish to use
    higher-order models at the leaves (e.g. quadratic, cubic).
    Example usage:
    \code{.cpp}
    reg = linxgb(n_estimators=5)
    reg.fit(make_polynomial_features(X_train,order=2),y_train)
    y_pred = reg.fit(make_polynomial_features(X_test,order=2))
    \endcode
    """

    assert isinstance(X, np.ndarray), "X must be a numpy ndarray!"

    poly = PolynomialFeatures(order)
    X = poly.fit_transform(X)
    X = X[:,1:] # remove the first column, only 1
    return X

class linxgb:
    """Define a LinXGBoost regressor.

    It basically holds a list of trees.
    Following the philosophy of <a href="http://scikit-learn.org/">sklearn</a>,
    two functions are exposed: fit() and predict().
    Example usage:
    \code{.cpp}
    reg = linxgb(n_estimators=5,lbda=0.,min_samples_leaf=3)
    reg.fit(X_train,y_train)
    y_pred = reg.fit(X_test)
    \endcode
    """

    def __init__(self, loss_function="reg:squarederror", n_estimators=5,
                 min_samples_split=3, min_samples_leaf=2, max_depth=6,
                 max_samples_linear_model=sys.maxsize,
                 subsample=1.0,
                 learning_rate=0.3, min_split_loss=0.0, gamma=0.0, lbda=0.0,
                 prune=True,
                 random_state=None,
                 verbose=0, nthread=1):

        self.loss_function = loss_function
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_samples_linear_model = max_samples_linear_model
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.min_split_loss = min_split_loss
        self.lbda = lbda
        self.gamma = gamma
        self.prune = prune
        self.random_state = random_state
        self.verbose = verbose
        self.nthread = nthread

        self.check_params()

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **params):
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        for key in params.keys():
            if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
            setattr(self, key, params[key])
        return self

    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if self.lbda < 0.0:
            raise ValueError("lbda must be greater than 0 but "
                             "was %r" % self.lbda)

        if self.gamma < 0.0:
            raise ValueError("gamma must be greater than 0 but "
                             "was %r" % self.gamma)

    def _predict(self, X):
        """Make predictions.

        This function is for internal use only.
        Users must call linxgb.predict().
        Why? At each stage, we add a new tree. To build this new tree, we need
        the predictions done the model made up of all trees built so far. The
        predictions of each tree is down-weighted by the learning rate.
        If the user calls this function once the full model is available,
        the last tree will also be down-weighted, which is non-sense.
        """

        n = X.shape[0]
        y = np.zeros(n, dtype=float)
        for tree in self.trees:
            y += self.learning_rate*tree.predict(X)

        return y

    def predict(self, X):
        """Make predictions.
        """

        n = X.shape[0]
        y = np.zeros(n, dtype=float)
        if not self.trees:
            return y
        for t in range(len(self.trees)-1):
            y += self.learning_rate*self.trees[t].predict(X)
        y += self.trees[-1].predict(X)
        
        # in binary classification, outputs >0 are labeled 1, 0 otherwise
        if self.loss_function == "binary:logistic":
            y[ y<=0 ] = 0
            y[ y>0 ] = 1
        
        return y

    def squareloss(self, y, y_hat):
        """Return the squared loss wo/ penalty / regularization.
        """

        return np.sum(np.square(y_hat-y))

    def dsquareloss(self, X, y, y_hat):
        """Return the first-order derivative of the squared loss
        w.r.t. its second argument evaluated at \f$(y, \hat{y}^{(t-1)})\f$.
        """

        return 2*(y_hat-y)
    
    def ddsquareloss(self, X, y, y_hat):
        """Return the second-order derivative of the squared loss
        w.r.t. its second argument evaluated at \f$(y, \hat{y}^{(t-1)})\f$.
        """
        
        n = len(y)
        return 2*np.ones(n, dtype=float)
    
    def logisticloss(self, y, y_hat):
        """Return the logisitc loss wo/ penalty / regularization.
        """
        
        return np.sum(y*np.log(1.+np.exp(-y_hat)) + (1.-y)*np.log(1.+np.exp(y_hat)))
        
    def dlogisticloss(self, X, y, y_hat):
        """Return the first-order derivative of the logistic loss
        w.r.t. its second argument evaluated at \f$(y, \hat{y}^{(t-1)})\f$.
        """
        
        return -( (y-1.)*np.exp(y_hat)+y)/(np.exp(y_hat)+1.)
        
    def ddlogisticloss(self, X, y, y_hat):
        """Return the second-order derivative of the logistic loss
        w.r.t. its second argument evaluated at \f$(y, \hat{y}^{(t-1)})\f$.
        """
        
        return np.exp(y_hat)/np.square(np.exp(y_hat)+1.)
        
    def regularization(self):
        """Return the penalty for all trees built so far.

        \f$\gamma\f$ penalizes the number of leaves and
        \f$\lambda\f$ penalizes the coefficients of the models at the leaves
        (except the intercept).
        """

        reg = 0.
        for tree in self.trees:
            reg += tree.regularization(gamma=self.gamma, lbda=self.lbda)

        return reg

    def objective(self,X, y, y_hat=None):
        if y_hat is None:
            y_hat = self._predict(X)
        return self.loss_func(y,y_hat)+self.regularization()

    def build_tree(self, tree, X, g, h):
        """Recursively build a tree.

        The first tree we pass is a leaf: node(verbose=self.verbose).
        Then for that leaf, we build the linear model.
        Thereafter, we investigate where to best slit the leaf.
        If a best split is found, and if the split is allowed, then a left child
        and a right child are created, and linxgb.build_tree() is called
        for the left child and the right child.
        """

        assert tree.left == None, "the node must be a leaf!"

        n, d = X.shape
        linear_model = ( n > d ) and ( n <= self.max_samples_linear_model )
        try:
            tree.set_weight(X, g, h, self.lbda, linear_model)
        except:
            print( "in tree building: something went wrong!" )
            raise
        if tree.depth >= self.max_depth:
            return tree
        if n < self.min_samples_split:
            return tree
        tree.find_best_split(X, g, h, self.lbda, self.gamma, self.max_samples_linear_model, self.min_samples_leaf)
        if tree.split_feature == -1: # no split because of the constraints
            if self.verbose > 2:
                print( "node could not be split" )
            return tree
        left_child, right_child = tree.add()
        c = ( X[:,tree.split_feature] < tree.split_value )
        if self.verbose > 1:
            print( "creating left child with {:d} instances".format(np.sum(c)) )
        try:
            self.build_tree(left_child, X[c,:], g[c], h[c])
        except RuntimeError:
            print( "maximum recursion depth exceeded: Tree depth is {}".format(tree.depth) )
            tree.left = None
            tree.right = None
            return tree
            raise

        c = np.invert(c)
        if self.verbose > 1:
            print( "creating right child with {:d} instances".format(np.sum(c)) )
        try:
            self.build_tree(right_child, X[c,:], g[c], h[c])
        except RuntimeError:
            print( "maximum recursion depth exceeded: Tree depth is {}".format(tree.depth) )
            tree.left = None
            tree.right = None
            return tree
            raise

        return tree

    def fit(self, X, y):
        """Fit the model by building all trees
        """
        
        if self.loss_function == "reg:linear":
            print("reg:linear will be deprecated; use reg:squarederror instead")
            self.loss_function = "reg:squarederror"
        if self.loss_function == "reg:squarederror":
            self.loss_func = self.squareloss
            self.dloss_func = self.dsquareloss
            self.ddloss_func = self.ddsquareloss
        elif self.loss_function == "binary:logistic":
            self.loss_func = self.logisticloss
            self.dloss_func = self.dlogisticloss
            self.ddloss_func = self.ddlogisticloss
        else:
            raise ValueError("unknown error function")
        
        if y.ndim != 1:
            print( "lingxb.fit() is expecting a 1D array!" )
            y = y.ravel()
        assert X.shape[0] == len(y)

        self.trees = []
        self.tree_objs = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.tree_objs.append( self.objective(X,y) )

        for t in range(0,self.n_estimators):
            n = X.shape[0]
            batch_size = int(np.rint(self.subsample*n))
            indices = np.random.choice(n, batch_size, replace=False)
            y_hat = self._predict(X[indices,:])
            g = self.dloss_func(X[indices,:],y[indices],y_hat)
            h = self.ddloss_func(X[indices,:],y[indices],y_hat)
            if self.verbose > 0:
                print( "building tree {}, total obj={}".format(t+1,np.sum(self.tree_objs)) )
            tree = self.build_tree( node(verbose=self.verbose), X[indices,:], g, h )
            self.trees.append(tree)
            tree_obj = tree.objective(self.gamma)
            if self.verbose > 0:
                print( "tree max. depth={}, num. of leaves={}, obj_{}={}, total obj={}". \
                       format(tree.max_depth(),tree.num_leaves(), t+1, tree_obj, np.sum(self.tree_objs)+tree_obj) )
            # pruning
            if self.prune:
                num_pruning = self.prune_tree_type_2(tree)
                if num_pruning > 0:
                    tree_obj = tree.objective(self.gamma)
                    if self.verbose > 0:
                        print("{} nodes pruned, ".format(num_pruning))
                        print( "tree max. depth={}, num. of leaves={}, obj_{}={}, total obj={}". \
                               format(tree.max_depth(),tree.num_leaves(), t+1, tree_obj, np.sum(self.tree_objs)+tree_obj) )
            self.tree_objs.append(tree_obj)

            # check if the objective of the tree is positive
            if tree_obj > 0.:
                if self.verbose > 0:
                    print( "the objective of tree {}/{} of depth {} is positive: obj = {:.4e}". \
                            format(t+1,self.n_estimators,tree.max_depth(),tree_obj) )
                del self.trees[-1]
                del self.tree_objs[-1]
                break
        
        return self
        
    def prune_tree_type_1(self, tree):
        """Prune a tree.

        Pruning is done as in XGBoost.
        This type of pruning is not used
        since linxgb.prune_tree_type_2() yields much better results.
        """

        num_pruning = 0

        if not tree.is_leaf():
            if not tree.left.is_leaf():
                num_pruning += self.prune_tree_type_1(tree.left)
            if not tree.right.is_leaf():
                num_pruning += self.prune_tree_type_1(tree.right)
            if tree.left.is_leaf() and tree.right.is_leaf():
                if tree.gain < 0.:
                    if self.verbose > 1:
                        print( "pruning at depth {}".format(tree.depth) )
                    tree.left = None
                    tree.right = None
                    num_pruning += 1

        return num_pruning

    def prune_tree_type_2(self, tree):
        """Prune a tree.

        In XGBoost, a tree is grown until the maximum depth is reached.
        Then nodes with a negative gain are pruned out in a bottom-up fashion.
        Why do we accept negative gains?
        In the middle of the tree construction,
        the gain might be negative, but then the following gains might be significant.
        This is reminiscent of the exploitation vs. exploration
        in many disciplines, e.g. Reinforcement Learning:
        The best long-term strategy may involve short-term sacrifices.
        However, all sacrifices are unlikely to be worth it.
        Thus, in LinXGBoost, we investigate all subtrees
        starting from nodes with a negative gain
        in a top-to-bottom fashion
        and the subtrees that do not lead to a decrease of the objective
        are pruned out.
        """

        num_pruning = 0

        if not tree.is_leaf():
            if tree.gain < 0.:
                if tree.obj < tree.objective(gamma=self.gamma):
                    # undo the split
                    tree.left = None
                    tree.right = None
                    num_pruning += 1
            if not tree.is_leaf():
                num_pruning += self.prune_tree_type_2(tree.left)
                num_pruning += self.prune_tree_type_2(tree.right)

        return num_pruning
