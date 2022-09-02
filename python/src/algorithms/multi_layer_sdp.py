import numpy as np
import torch
import seaborn
import torch.nn as nn
import cvxpy as cp
from itertools import combinations
from matplotlib import pyplot as plt
import pylab as pyl
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import block_diag


class MultiLayerNN(nn.Module):


    def __init__(self, weights, bias_vecs):
        super(MultiLayerNN, self).__init__()
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.init_weights(weights, bias_vecs)
        self.in_dim = self.layers[0].weight.data.shape[1]


    def __str__(self):
        s = ""
        for i, l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                s += f"W{int(i/2)}: {l.weight.data} \n"
                s += f"b{int(i/2)} : {l.bias.data} \n"
            if isinstance(l, nn.modules.activation.ReLU):
                s += f"ReLU\n"
        return s


    def verify_weight_dims(self, weights, bias_vecs):
        assert len(weights) == len(bias_vecs)
        for i in range(0, len(weights)):
            _out_dim = weights[i].shape[0]
            if i < len(weights) - 1:
                assert _out_dim == len(bias_vecs[i+1])
            if i > 0:
                assert weights[i].shape[1] == weights[i-1].shape[0]
        return None


    def list_to_np(self, lmats):
        _m = []
        for m in lmats:
            _m.append(np.matrix(m))
        return _m


    def init_weights(self, weights, bias_vecs):
        weights = self.list_to_np(weights)
        self.verify_weight_dims(weights, bias_vecs)

        with torch.no_grad():
            for i, w in enumerate(weights):
                l = nn.Linear(w.shape[1], w.shape[0])
                l.weight.copy_(torch.tensor(w))
                l.bias.copy_(torch.tensor(bias_vecs[i]))
                self.layers.append(l)
                if i != len(weights)-1:
                    self.layers.append(self.relu)
        return


    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def build_M_out(S, weights, bias_vecs):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = _build_E(weights, 0)
    El = _build_E(weights, len(weights)-1)
    _out_ = np.block([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [weights[-1]*El,             np.matrix(bias_vecs[-1]).T],
        [np.zeros((1, E0.shape[1])), np.eye(1)]
    ])
    return _out_.T * S * _out_


def build_M_in(P, weights, bias_vecs):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)
    assert(P.shape[0] == weights[0].shape[1] + 1)
    E0 = _build_E(weights, 0)
    _in_ = cp.bmat([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [np.zeros((1, E0.shape[1])), np.eye(1)],
    ])
    M_in_P = _in_.T @ P @ _in_
    return M_in_P


def _build_T(dim, constraints=[]):
    T = cp.bmat(np.zeros((dim, dim)))
    index_combos = list(combinations(range(0, dim), 2))
    lambdas_ij = cp.Variable(len(index_combos), name='lambda_ij')
    constraints += [lambdas_ij >= 0]
    for k, (i,j) in enumerate(index_combos):
        ei = np.zeros((dim, 1))
        ei[i] = 1
        ej = np.zeros((dim, 1))
        ej[j] = 1
        v = cp.bmat([ei - ej])
        T += lambdas_ij[k] * (v @ v.T)
    return T, constraints


def _build_Q_for_relu(dim, constraints=[]):
    # build quadratic relaxation matrix for the graph of ReLU
    # applied componentwise on a vector in R^dim

    T, _constraints = _build_T(dim, constraints)
    constraints += _constraints

    alpha, beta = 0, 1
    _lambdas = cp.Variable(dim, 'lambda')
    _nus = cp.Variable(dim, name='nu')
    _etas = cp.Variable(dim, name='eta')

    constraints += [_nus >= 0]
    constraints += [_etas >= 0]

    nus = cp.bmat([_nus]).T
    etas = cp.bmat([_etas]).T
    diag_lambda = cp.diag(_lambdas)

    Q11 = -2 * alpha * beta * (diag_lambda + T)
    Q12 = (alpha + beta) * (diag_lambda + T)
    Q13 = -1 * (beta * nus) - (alpha * etas)
    Q22 = -2 * (diag_lambda + T)
    Q23 = nus + etas
    Q33 = np.zeros((1,1))

    Q = cp.bmat([
        [Q11,   Q12,   Q13],
        [Q12.T, Q22,   Q23],
        [Q13.T, Q23.T, Q33],
    ])
    return Q, constraints


def build_M_mid(Q, constraints, weights, bias_vecs, activ_func='relu'):
    assert len(weights) == len(bias_vecs)

    _m = sum([w.shape[0] for w in weights[:-1]])
    A = np.block([
        [block_diag(*weights[0:-1]), np.zeros((_m,weights[-1].shape[1]))]
    ])
    _m = sum([w.shape[0] for w in weights[1:]])
    B = np.block([
        [np.zeros((_m, weights[0].shape[1])), block_diag(*[np.eye(w.shape[1]) for w in weights[1:]])]
    ])
    bias_concat = np.matrix(np.concatenate(bias_vecs[:-1])).T

    _mid_ = np.block([
        [A,                         bias_concat],
        [B,                         np.zeros((B.shape[0], 1))],
        [np.zeros((1, B.shape[1])), np.eye(1,1)]
    ])

    M_mid_Q = _mid_.T @ Q @ _mid_
    return M_mid_Q, constraints


def _relaxation_for_hypercube(x, epsilon, constraints=[]):
    n = x.shape[0]
    x_floor = x - epsilon * np.ones((n,1))
    x_ceil =  x + epsilon * np.ones((n,1))

    a = cp.Variable(n, name='a')
    Gamma = cp.diag(a)
    constraints += [a >= 0]
    P = cp.bmat([
        [-2*Gamma,                     Gamma @ (x_floor + x_ceil)],
        [(x_floor + x_ceil).T @ Gamma, -2 * x_floor.T @ Gamma @ x_ceil]
    ])
    return P, constraints


def _relaxation_for_half_space(c, d, dim_x):
    # for half space defined by {y : cy < d} (in the output space of NN)
    # dim_x is the input space dimension of the NN
    dim_c = c.shape[0]
    S = np.block([
        [np.zeros((dim_x, dim_x)), np.zeros((dim_x, dim_c)), np.zeros((dim_x,1))],
        [np.zeros((dim_c, dim_x)), np.zeros((dim_c, dim_c)), c],
        [np.zeros((1, dim_x)),     c.T,                      -2*np.matrix([d])]
    ])
    return S


def _build_E(weights, k):
    E = []
    r = weights[k].shape[1]
    if k > 0:
        E.append(np.zeros((r, sum([w.shape[1] for w in weights[:k]]))))
    E.append(np.eye(r))
    if k < len(weights)-1:
        E.append(np.zeros((r, sum([w.shape[1] for w in weights[k+1:]]))))
    return np.matrix(np.block([ E ]))


def multi_layer_verification(x=[[1],[1]], eps=1, f=None):

    weights, bias_vecs = get_weights_from_nn(f)

    x = np.matrix(x)

    im_x = f(torch.tensor(x).T.float()).data.T.tolist()
    x_class = np.argmax(im_x)

    # TODO: make this work for higher dimensions (multiclass classifier)
    d = 0
    c = np.matrix('-1; 1') # defines halfspace where y1 > y2
    if x_class == 1:
        c = -1 * c # defines halfspace where y2 > y1

    P, constraints = _relaxation_for_hypercube(x=x, epsilon=eps)
    # TODO: verify what the shape of Q should actually be
    dim = sum([w.shape[0] for w in weights[:-1]]) # not sure this is correct
    Q, constraints = _build_Q_for_relu(dim=dim, constraints=constraints)
    S = _relaxation_for_half_space(c=c, d=d, dim_x=weights[0].shape[1])

    M_in_P = build_M_in(P, weights, bias_vecs)
    M_mid_Q, constraints = build_M_mid(Q=Q, constraints=constraints,
                                       weights=weights, bias_vecs=bias_vecs)
    M_out_S = build_M_out(S, weights, bias_vecs)
    X = M_in_P + M_mid_Q + M_out_S

    constraints += [X << 0]

    prob = cp.Problem(cp.Minimize(1), constraints)
    prob.solve()

    print(f"f({x.T}) = {im_x} |--> class: {x_class}")
    status = prob.status
    if status == cp.OPTIMAL:
        print(f"SUCCESS: all x within {eps} inf-norm of {x.T} are classified as class {x_class}")
    elif status == cp.INFEASIBLE:
        # How to check if this is a false negative? (maybe the relaxations aren't tight enough)
        print(f"COULD NOT verify all x within {eps} inf-norm of {x.T} are classified as {x_class}")
    elif status == cp.UNBOUNDED:
        print(f"Problem is unbounded - {cp.OPTIMAL_INACCURATE}")
    elif status == cp.OPTIMAL_INACCURATE:
        print(f"RUH ROH - {cp.OPTIMAL_INACCURATE}")
        print(f"SUCCESS?: all x within {eps} inf-norm of {x.T} are classified as class {x_class}")
    elif status == cp.INFEASIBLE_INACCURATE:
        print(f"RUH ROH - {cp.INFEASIBLE_INACCURATE}")
    elif status == cp.UNBOUNDED_INACCURATE:
        print(f"RUH ROH - {cp.UNBOUNDED_INACCURATE}")
    elif status == cp.INFEASIBLE_OR_UNBOUNDED:
        print(f"RUH ROH - {cp.INFEASIBLE_OR_UNBOUNDED}")

    # print("P = \n", P.value)
    # print("Q = \n", Q.value)


def get_weights_from_nn(f):
    # only handles 'flat' ffnn's (for now)
    # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    weights, bias_vecs = [], []
    for i, l in enumerate(f.layers):
        if isinstance(l, nn.modules.linear.Linear):
            weights.append(l.weight.data.numpy())
            bias_vecs.append(l.bias.data.numpy())
        else:
            assert isinstance(l, nn.modules.activation.ReLU)
    return weights, bias_vecs


if __name__ == '__main__':

    # TODO: Setup testing infra
    # TODO: try with more layers
    # TODO: try with non-square matrices
    # TODO: handle optimal_inaccurate (numerical instability?)
    weights = [
        [[1, 0],
         [0, 1]],
        [[2, 0],
         [0, 2]],
        [[3, 0],
         [0, 3]]
    ]
    bias_vecs =[
        (1,1),
        (2,2),
        (3,3),
    ]
    f = MultiLayerNN(weights, bias_vecs)
    # print(f)
    multi_layer_verification(x=[[9],[1]], eps=0.007, f=f)
