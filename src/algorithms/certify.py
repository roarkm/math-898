from itertools import combinations
import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
import sympy as sp
from sympy import BlockDiagMatrix
from scipy.linalg import block_diag
from src.models.multi_layer import MultiLayerNN
from src.algorithms.abstract_verifier import AbstractVerifier


class Certify(AbstractVerifier):

    def __init__(self, f=None):
        super(Certify, self).__init__(f)
        self.constraints = []


    def __str__(self):
        s = 'Certify Algorithm\n'
        s += super().__str__()
        return s


    def build_symbolic_matrices(self, x=None, eps=1):
        x = np.array(x)
        im_x = self.f(torch.tensor(x).T.float()).data.T.tolist()
        x_class = np.argmax(im_x)

        d = 0
        c = self.sep_hplane_for_advclass(x, complement=True)

        P = symbolic_relaxation_for_hypercube(x=x, epsilon=eps)
        # sp.pprint(P)
        dim = sum([w.shape[0] for w in self.nn_weights[:-1]])
        Q = symbolic_relaxation_for_relu(dim=dim)
        # sp.pprint(Q)
        S = symbolic_relaxation_for_half_space(c=c, d=d, dim_x=self.nn_weights[0].shape[1])
        # sp.pprint(S)

        M_in_P = symbolic_build_M_in(P, self.nn_weights, self.nn_bias_vecs)
        # sp.pprint(M_in_P)
        M_mid_Q = symbolic_build_M_mid(Q=Q, weights=self.nn_weights, bias_vecs=self.nn_bias_vecs)
        # sp.pprint(M_mid_Q)
        exit()
        # M_out_S = build_M_out(S, self.nn_weights, self.nn_bias_vecs)

        # X = M_in_P + M_mid_Q + M_out_S

    def verifiy_at_point(self, x=[[9],[0]], eps=1, verbose=False, max_iters=10**6):

        x = np.array(x)
        im_x = self.f(torch.tensor(x).T.float()).data.T.tolist()
        x_class = np.argmax(im_x)

        d = 0
        c = self.sep_hplane_for_advclass(x, complement=True)

        P, constraints = _relaxation_for_hypercube(x=x, epsilon=eps)
        dim = sum([w.shape[0] for w in self.nn_weights[:-1]])
        Q, constraints = _build_Q_for_relu(dim=dim, constraints=constraints)
        S = _relaxation_for_half_space(c=c, d=d, dim_x=self.nn_weights[0].shape[1])

        M_in_P = build_M_in(P, self.nn_weights, self.nn_bias_vecs)
        M_mid_Q, constraints = build_M_mid(Q=Q, constraints=constraints,
                                           weights=self.nn_weights, bias_vecs=self.nn_bias_vecs)
        M_out_S = build_M_out(S, self.nn_weights, self.nn_bias_vecs)

        X = M_in_P + M_mid_Q + M_out_S

        constraints += [X << 0]

        prob = cp.Problem(cp.Minimize(1), constraints)
        prob.solve(verbose=verbose, max_iters=max_iters, solver=cp.CVXOPT)

        debug = ""
        debug += f"f({x.T}) = {im_x} |--> class: {x_class}\n"
        debug += f"{prob.status}\n"
        status = prob.status
        if status == cp.OPTIMAL:
            debug += f"SUCCESS: all x within {eps} inf-norm of {x.T} are classified as class {x_class}\n"
            verified = True
        elif status == cp.OPTIMAL_INACCURATE:
            debug += f"SUCCESS?: all x within {eps} inf-norm of {x.T} are classified as class {x_class}\n"
            verified = True
        elif status == cp.INFEASIBLE:
            # How to check if this is a false negative? (maybe the relaxations aren't tight enough)
            debug += f"COULD NOT verify all x within {eps} inf-norm of {x.T} are classified as {x_class}\n"
            verified = False
        elif status == cp.INFEASIBLE_INACCURATE:
            debug += f"COULD NOT verify all x within {eps} inf-norm of {x.T} are classified as {x_class}\n"
            verified = False
        elif status == cp.UNBOUNDED:
            debug += f"COULD NOT verify all x within {eps} inf-norm of {x.T} are classified as {x_class}\n"
            verified = False
        elif status == cp.UNBOUNDED_INACCURATE:
            debug += f"COULD NOT verify all x within {eps} inf-norm of {x.T} are classified as {x_class}\n"
            verified = False
        elif status == cp.INFEASIBLE_OR_UNBOUNDED:
            debug += f"COULD NOT verify all x within {eps} inf-norm of {x.T} are classified as {x_class}\n"
            verified = False

        if verbose:
            if verified:
                debug += f"P =\n {P.value}\n"
                debug += f"Q =\n {Q.value}\n"
            print(debug)
        return verified

def build_M_out(S, weights, bias_vecs):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = _build_E(weights, 0)
    El = _build_E(weights, len(weights)-1)
    _out_ = np.block([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [weights[-1]@El,             np.array([bias_vecs[-1]]).T],
        [np.zeros((1, E0.shape[1])), np.eye(1)]
    ])
    return _out_.T @ S @ _out_


def symbolic_build_M_in(P, weights, bias_vecs):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)
    assert(P.shape[0] == weights[0].shape[1] + 1)
    E0 = _build_E(weights, 0)
    E0 = sp.Matrix(E0)
    _in_ = sp.BlockMatrix([
        [E0,                       sp.zeros(E0.shape[0], 1)],
        [sp.zeros(1, E0.shape[1])               , sp.eye(1)],
    ])
    M_in_P = _in_.T @ P @ _in_
    return M_in_P


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


def symbolic_build_M_mid(Q, weights, bias_vecs):
    assert len(weights) == len(bias_vecs)

    _m = sum([w.shape[0] for w in weights[:-1]])

    sp_weights = [sp.Matrix(w) for w in weights]
    A = sp.BlockMatrix([
        [BlockDiagMatrix(*sp_weights[0:-1]), sp.zeros(_m,weights[-1].shape[1])]
    ])
    _m = sum([w.shape[0] for w in weights[1:]])
    B = sp.BlockMatrix([
        [sp.zeros(_m, weights[0].shape[1]), BlockDiagMatrix(*[sp.eye(w.shape[1]) for w in weights[1:]])]
    ])
    bias_concat = np.array([np.concatenate(bias_vecs[:-1])]).T
    _mid_ = sp.BlockMatrix([
        [A,                        sp.Matrix(bias_concat)],
        [B,                       sp.zeros(B.shape[0], 1)],
        [sp.zeros(1, B.shape[1]),             sp.eye(1,1)]
    ])

    M_mid_Q = _mid_.T @ Q @ _mid_
    return M_mid_Q


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
    bias_concat = np.array([np.concatenate(bias_vecs[:-1])]).T

    _mid_ = np.block([
        [A,                         bias_concat],
        [B,                         np.zeros((B.shape[0], 1))],
        [np.zeros((1, B.shape[1])), np.eye(1,1)]
    ])

    M_mid_Q = _mid_.T @ Q @ _mid_
    return M_mid_Q, constraints


def symbolic_build_T(dim):
    T = sp.zeros(dim, dim)
    lambdas_ij = [] # keep track of free sympy variables
    for (i,j) in combinations(range(0, dim), 2):
        l = sp.symbols(f"lambda{i}{j}")
        lambdas_ij.append(l)
        ei = sp.zeros(dim, 1)
        ei[i] = 1
        ej = sp.zeros(dim, 1)
        ej[j] = 1
        v = ei - ej
        m = l * sp.MatMul(v, v.T)
        T += m
    return T, lambdas_ij


def symbolic_relaxation_for_relu(dim):
    # build quadratic relaxation matrix for the graph of ReLU
    # applied componentwise on a vector in R^dim

    T, Q_vars = symbolic_build_T(dim)
    # using full generallity in case I want to extend to other activations
    alpha, beta = 0, 1
    lambdas = [sp.symbols(f"lambda{i}") for i in range(0, dim)]
    nus = sp.Matrix([[sp.symbols(f"nu{i}")] for i in range(0, dim)])
    etas = sp.Matrix([[sp.symbols(f"eta{i}")] for i in range(0, dim)])

    # keep track of free sympy variables
    Q_vars = Q_vars + [nu for nu in nus]
    Q_vars = Q_vars + lambdas
    Q_vars = Q_vars + [nu for nu in nus]

    diag_lambda = sp.diag(*lambdas)

    Q11 = -2 * alpha * beta * (diag_lambda + T)
    Q12 = (alpha + beta) * (diag_lambda + T)
    Q13 = -1 * beta * nus - alpha * etas
    Q22 = -2 * (diag_lambda + T)
    Q23 = nus + etas
    Q33 = sp.zeros(1,1)

    Q = sp.BlockMatrix([
        [Q11,   Q12,   Q13],
        [Q12.T, Q22,   Q23],
        [Q13.T, Q23.T, Q33],
    ])
    assert(sp.Matrix(Q).is_symmetric())
    return Q


def symbolic_relaxation_for_hypercube(x, epsilon):
    n = x.shape[0]
    x_floor = x - epsilon * np.ones((n,1))
    x_ceil =  x + epsilon * np.ones((n,1))

    _g = ' '.join([f"g{i}" for i in range(1, n+1)])
    g = sp.var(_g)
    Gamma = sp.diag(*g)
    P = sp.Matrix([
        [-2 * Gamma,                        Gamma @ (x_floor + x_ceil)],
        [(x_floor + x_ceil).T @ Gamma, -2 * x_floor.T @ Gamma @ x_ceil]
    ])
    return P


def _relaxation_for_hypercube(x, epsilon, constraints=[]):
    n = x.shape[0]
    x_floor = x - epsilon * np.ones((n,1))
    x_ceil =  x + epsilon * np.ones((n,1))

    a = cp.Variable(n, name='a')
    Gamma = cp.diag(a)
    constraints += [a >= 0]
    P = cp.bmat([
        [-2 * Gamma,                   Gamma @ (x_floor + x_ceil)],
        [(x_floor + x_ceil).T @ Gamma, -2 * x_floor.T @ Gamma @ x_ceil]
    ])
    return P, constraints


def symbolic_relaxation_for_half_space(c, d, dim_x):
    # for half space defined by {y : cy < d} (in the output space of NN)
    # dim_x is the input space dimension of the NN
    dim_c = c.shape[0]
    S = np.block([
        [np.zeros((dim_x, dim_x)), np.zeros((dim_x, dim_c)), np.zeros((dim_x,1))],
        [np.zeros((dim_c, dim_x)), np.zeros((dim_c, dim_c)), c],
        [np.zeros((1, dim_x)),     c.T,                      -2*np.array([[d]])]
    ])
    return S
    # sympy likely not needed but the following does work
    c = sp.Matrix([c])
    S = sp.BlockMatrix([
        [sp.zeros(dim_x, dim_x), sp.zeros(dim_x, dim_c), sp.zeros(dim_x,1)],
        [sp.zeros(dim_c, dim_x), sp.zeros(dim_c, dim_c), c.T],
        [sp.zeros(1, dim_x),     c,                      -2*sp.Matrix([[d]])]
    ])
    return S


def _relaxation_for_half_space(c, d, dim_x):
    # for half space defined by {y : cy < d} (in the output space of NN)
    # dim_x is the input space dimension of the NN
    dim_c = c.shape[0]

    S = np.block([
        [np.zeros((dim_x, dim_x)), np.zeros((dim_x, dim_c)), np.zeros((dim_x,1))],
        [np.zeros((dim_c, dim_x)), np.zeros((dim_c, dim_c)), c],
        [np.zeros((1, dim_x)),     c.T,                      -2*np.array([[d]])]
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
    return np.array(np.block([ E ]))


if __name__ == '__main__':

    # TODO: Setup testing infra
    # TODO: try with non-square matrices
    weights = [
        [[1, 0],
         [0, 1]],
        [[1, 0],
         [0, 1]],
        # [[3, 0],
         # [0, 3]],
        # [[4, 0],
         # [0, 4]],
    ]
    bias_vecs =[
        (0,0),
        (0,0),
        # (3,3),
        # (4,4),
    ]
    f = MultiLayerNN(weights, bias_vecs)
    cert = Certify(f)
    eps = 0.5
    x = [[9], [0]]
    cert.build_symbolic_matrices(x=x, eps=eps)
    exit()
    # is_robust = cert.verifiy_at_point(x=x, eps=eps)
    # print(f"Identity is {eps}-robust at {x}? {is_robust}")
    # x = [[0], [9]]
    # is_robust = cert.verifiy_at_point(x=x, eps=eps)
    # print(f"Identity is {eps}-robust at {x}? {is_robust}")
