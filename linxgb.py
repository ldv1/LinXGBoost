import sys
import numpy as np
from node import node

def make_quadratic(X):
    assert isinstance(X, np.ndarray), "X must be a numpy ndarray!"
    return np.c_[ X, X*X ]

class linxgb:
    def __init__(self, loss_func="square_loss", n_estimators=5,
                 min_samples_split=3, min_samples_leaf=2, max_depth=6,
                 max_samples_linear_model=sys.maxsize,
                 subsample=1.0,
                 learning_rate=0.3, min_split_loss=0.0, gamma=0.0, lbda=0.0,
                 prune=True,
                 random_state=None,
                 verbose=0, nthread=1):

        self.loss_func = loss_func
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
        n = X.shape[0]
        y = np.zeros(n, dtype=float)
        for tree in self.trees:
            y += self.learning_rate*tree.predict(X)

        return y

    def predict(self, X):
        n = X.shape[0]
        y = np.zeros(n, dtype=float)
        if not self.trees:
            return y
        for t in range(len(self.trees)-1):
            y += self.learning_rate*self.trees[t].predict(X)
        y += self.trees[-1].predict(X)

        return y

    def loss(self, y, y_hat):
        return np.sum(np.square(y_hat-y))

    def dloss(self, X, y, y_hat):
        n = len(y)
        return 2*(y_hat-y), 2*np.ones(n, dtype=float)

    def regularization(self):
        reg = 0.
        for tree in self.trees:
            reg += tree.regularization(gamma=self.gamma, lbda=self.lbda)

        return reg

    def objective(self,X, y, y_hat=None):
        if y_hat is None:
            y_hat = self._predict(X)
        return self.loss(y,y_hat)+self.regularization()

    def build_tree(self, tree, X, g, h):
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
            g, h = self.dloss(X[indices,:],y[indices],y_hat)
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

    def prune_tree_type_1(self, tree):
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
