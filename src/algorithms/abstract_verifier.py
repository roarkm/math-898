import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn


class AbstractVerifier():

    def __init__(self, f=None):
        self.f = f
        self.constraints = []
        self.relu = nn.ReLU()
        if f:
            self.nn_weights, self.nn_bias_vecs = weights_from_nn(self.f)


    def str_constraints(self):
        if len(self.constraints) == 0:
            return "No Constraints."
        s = ""
        for c in self.constraints:
            s += f"\nconstraint:"
            s += str(c)
        return s


    def __str__(self):
        s = ''
        if not self.f:
            s += "No nn provided."
        else:
            s += f"f:R^{self.nn_weights[0].shape[1]} -> R^{self.nn_weights[-1].shape[0]}"
            s += f"\t{len(self.nn_weights)} layers"
        return s


    def sep_hplane_for_advclass(self, x, complement=False):
        fx = self.f(torch.tensor(np.array(x)).T.float()).detach().numpy()
        class_order = np.argsort(fx)[0]
        # index of component with largest value
        x_class = class_order[-1]
        # index of component with second largest value
        adversarial_class = class_order[-2]
        return _vector_for_separating_hyperplane(large_index=x_class,
                                                 small_index=adversarial_class,
                                                 n=fx.shape[1],
                                                 complement=complement)


def constraints_for_separating_hyperplane(opt_vars, large_index, small_index,
                                          verbose=False, complement=False):
    # create constraint for a seperating hyperplane in R^n
    # where n = dim(opt_vars)
    assert large_index <= opt_vars.shape[1]
    assert small_index <= opt_vars.shape[1]

    c = _vector_for_separating_hyperplane(large_index, small_index, opt_vars.shape[1])
    if complement:
        return [c @ opt_vars <= 0]
    return [c @ opt_vars >= 0]


def constraints_for_k_class_polytope(k, x, verbose=False, complement=False):
    # build a polytope constraint matrix corresponding to the output region
    # of the k-th class
    # ie - the polytope corresponding to the region of R^dim where
    # the k-th component is greater than all other components
    # returns a list containing the inequality constraints in matrix form
    assert complement == False, "Complement option not yet supported"
    z = cp.bmat([cp.Variable(len(x), name='z')]).T
    n_rows = len(x)-1
    A = np.zeros((n_rows, len(x)))
    A[:,k] = 1

    row = 0
    for j in range(0, len(x)):
        if j == k:
            continue
        A[row][j] = -1
        row += 1

    b = np.zeros((n_rows, 1))
    if verbose:
        print(f"Polytope for {len(x)}-class classifier. k={k}.")
        print(A)
        print(b)
    return [A @ z >= b]


def weights_from_nn(f):
    """
    :type  : MultiLayerNN
    :rtype : tuple(list[np.array], list[np.array])
    """
    # only handles 'flat' ffnn's (for now)
    # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    weights, bias_vecs = [], []
    for i, l in enumerate(f.layers):
        if isinstance(l, nn.modules.linear.Linear):
            weights.append(l.weight.data.numpy())
            bias_vecs.append(l.bias.data.numpy())
        else:
            assert isinstance(l, nn.modules.activation.ReLU)
    assert len(weights) >= 2, "Only supporting more than two layers"
    return weights, bias_vecs


def _vector_for_separating_hyperplane(large_index, small_index, n, complement=False):
    # create vector c for a seperating hyperplane
    #    {x | c dot x >= 0 and x,c in R^n}
    c = np.zeros((1, n))
    c[0][large_index] = 1
    c[0][small_index] = -1
    if complement:
        return -1 * c.T
    return c.T


