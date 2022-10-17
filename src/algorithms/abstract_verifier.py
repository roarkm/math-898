import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn


class AbstractVerifier():

    def __init__(self, f=None):
        self.f = f
        self.constraints = []
        self._free_vars = []
        self.relu = nn.ReLU()
        if f:
            self.nn_weights, self.nn_bias_vecs = weights_from_nn(self.f)

    def add_free_var(self, var):
        self._free_vars.append(var)

    def free_vars(self, var_name=None, names_only=False):
        if names_only:
            if self._free_vars == []:
                return []
            return [v.name() for v in self._free_vars]
        if var_name is None:
            return self._free_vars
        return [v for v in self._free_vars if v.name() == var_name][0]

    def str_constraints(self):
        return str_constraints(self.constraints)

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


def _vector_for_separating_hyperplane(large_index, small_index, n, complement=False):
    # create vector c for a seperating hyperplane
    #    {x | c dot x >= 0 and x,c in R^n} = {x | x_large_index > x_small_index }
    c = np.zeros((1, n))
    c[0][large_index] = 1
    c[0][small_index] = -1
    if complement:
        return -1 * c.T
    return c.T


def constraints_for_separating_hyperplane(opt_vars, large_index, small_index,
                                          verbose=False, complement=False):
    # create constraint for a seperating hyperplane in R^n
    # where n = dim(opt_vars)
    assert large_index <= opt_vars.shape[1]
    assert small_index <= opt_vars.shape[1]

    c = _vector_for_separating_hyperplane(large_index, small_index,
                                          opt_vars.shape[1],
                                          complement=complement)
    return [c @ opt_vars >= np.zeros((1,1))]


def mat_for_k_class_polytope(k, dim, complement=False):
    assert complement == False, "Complement option not yet supported"
    if dim == 1:
        return np.zeros((1,1)), np.zeros((1,1))

    n_rows = dim-1
    A = np.zeros((n_rows, dim))
    A[:,k] = 1

    row = 0
    for j in range(0, dim):
        if j == k:
            continue
        A[row][j] = -1
        row += 1
    b = np.zeros((n_rows, 1))
    return (A, b)


def constraints_for_k_class_polytope(k, x, verbose=False, complement=False):
    # build a polytope constraint matrix corresponding to the output region
    # of the k-th class
    # ie - the polytope corresponding to the region of R^dim where
    # the k-th component is greater than all other components
    # returns a list containing the inequality constraints in matrix form
    z = cp.bmat([cp.Variable(len(x), name='z')]).T
    A, b = mat_for_k_class_polytope(k, len(x), complement)
    return [A @ z >= b]


def constraints_for_inf_ball(center, eps, free_vars=None, free_vars_name=None):
    # create a list containing the constraints for an inf-ball of radius eps
    if free_vars is not None:
        assert free_vars.shape[0] == len(center)
    else:
        if free_vars_name == None:
            free_vars_name = 'z0'
        free_vars = cp.Variable((len(center),1), name=free_vars_name)

    A_lt = np.zeros((2, len(center)))
    b_vec = np.zeros((2, 1))

    # top constraint
    A_lt[0][0] = 1
    b_vec[0][0] = center[0][0] + eps
    # bottom constraint
    A_lt[1][0] = -1
    b_vec[1][0] = -1 * (center[0][0] - eps)

    if len(center) > 1:
        for i in range(1, len(center)):
            # top constraint
            row = np.zeros((1, len(center)))
            row[0][i] = 1
            A_lt = np.vstack([A_lt, row])
            _a = np.matrix([[ float(center[i][0]) + eps ]])
            b_vec = np.vstack([b_vec, _a])

            # bottom constraint
            row = np.zeros((1, len(center)))
            row[0][i] = -1
            A_lt = np.vstack([A_lt, row])
            _a = np.matrix([[ -1 * (float(center[i][0]) - eps) ]])
            b_vec = np.vstack([b_vec, _a])
    return [A_lt @ free_vars <= b_vec], free_vars


def str_constraints(constraints):
    if len(constraints) == 0:
        return "No Constraints."
    s = ""
    for c in constraints:
        s += f"\nconstraint:\n"
        s += str(c)
    return s


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


