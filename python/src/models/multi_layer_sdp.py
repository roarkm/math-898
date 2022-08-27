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

class TwoLayerFFNet(nn.Module):
    # A simple 1 layer feedforward nn with relu activation
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerFFNet, self).__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, output_dim)


    def __str__(self):
        # s = super(TwoLayerFFNet, self).__str__()
        s = ""
        s += f"w0 : {self.fc0.weight.data} \n"
        s += f"b0 : {self.fc0.bias.data} \n"
        s += f"w1 : {self.fc1.weight.data} \n"
        s += f"b1 : {self.fc1.bias.data} \n"
        return s


    def init_weights(self, w0=None, b0=None, w1=None, b1=None):
        with torch.no_grad():
            if w0:
                self.fc0.weight.copy_(torch.tensor(w0))
            if b0:
                self.fc0.bias.copy_(torch.tensor(b0))
            if w1:
                self.fc1.weight.copy_(torch.tensor(w1))
            if b1:
                self.fc1.bias.copy_(torch.tensor(b1))
        return


    def forward(self, x):
        out = self.fc0(x)
        out = self.relu(out)
        out = self.fc1(out)
        return out


def build_M_out(S, weights, bias_vecs):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = _build_E(weights, 0)
    El = _build_E(weights, len(weights)-1)

    _out_ = np.block([
        [E0,             np.zeros((E0.shape[0], 1))],
        [weights[-1]*El, np.matrix(bias_vecs[-1]).T],
        [np.zeros((1, E0.shape[1])), np.eye(1)]
    ])
    return _out_.T * S * _out_


def build_M_in(P, weights, bias_vecs):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)
    assert(P.shape[0] == weights[0].shape[1] + 1)
    E0 = _build_E(weights, 0)
    _in_ = cp.bmat([
        [E0, np.zeros((E0.shape[0], 1))],
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


def verify_weight_dims(weights, bias_vecs):
    assert len(weights) == len(bias_vecs)
    for i in range(0, len(weights)-1):
        last_out_dim = weights[i].shape[0]
        assert last_out_dim == bias_vecs[i].shape[0]
        if i > 0:
            assert weights[i].shape[1] == last_out_dim
    return None


def _build_E(weights, k):
    E = []
    r = weights[k].shape[1]
    if k > 0:
        E.append(np.zeros((r, sum([w.shape[1] for w in weights[:k]]))))
    E.append(np.eye(r))
    if k < len(weights)-1:
        E.append(np.zeros((r, sum([w.shape[1] for w in weights[k+1:]]))))
    return np.matrix(np.block([ E ]))


def multi_layer_verification(x=[[9],[1]], eps=1, f=None):

    weights, bias_vecs = get_weights_from_nn(f)
    verify_weight_dims(weights, bias_vecs)

    x = np.matrix(x)
    xl = _build_xl(x, weights, bias_vecs)
    # f_in_dim = weights[0].shape[0]

    im_x = f(torch.tensor(x).T.float()).data.T.tolist()
    x_class = np.argmax(im_x)

    # TODO: make this work for higher dimensions
    d=0
    c = np.matrix('-1; 1') # defines halfspace where y1 > y2
    if x_class == 1:
        c = -1 * c # defines halfspace where y2 > y1

    P, constraints = _relaxation_for_hypercube(x=x, epsilon=eps)
    Q, constraints = _build_Q_for_relu(dim=weights[0].shape[0], constraints=constraints)
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
    # TODO: what do these mean (in this context)?
    # elif status == cp.UNBOUNDED:
    # elif status == cp.OPTIMAL_INACCURATE:
    # elif status == cp.INFEASIBLE_INACCURATE:
    # elif status == cp.UNBOUNDED_INACCURATE:
    # elif status == cp.INFEASIBLE_OR_UNBOUNDED:

    # print("P = \n", P.value)
    # print("Q = \n", Q.value)


def get_weights_from_nn(f):
    # only handles 'flat' ffnn's (for now)
    # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    weights, bias_vecs = [], []
    for i, c in enumerate(f.children()):
        if isinstance(c, nn.modules.linear.Linear):
            weights.append(c.weight.data.numpy())
            bias_vecs.append(c.bias.data.numpy())
        else:
            assert isinstance(c, nn.modules.activation.ReLU)
    return weights, bias_vecs


def _build_xl(x0, weights, bias):
    _xi = np.matrix(x0)
    xl = _xi
    for i, w in enumerate(weights):
        _xj = (np.matrix(w) * _xi) + np.matrix(bias[i]).T
        xl = np.concatenate((xl, _xj))
        _xi = _xj
    return xl


if __name__ == '__main__':

    W0 = [[1, 0],
          [0,1]]
    b0 = (0,0)
    W1 = [[.9, 0],
          [0,.9]]
    b1 = (0,0)
    f = TwoLayerFFNet(2,2,2)
    f.init_weights(w0=W0, b0=b0, w1=W1, b1=b1)
    multi_layer_verification(x=[[1],[1]], eps=0.007, f=f)
