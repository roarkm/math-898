from itertools import combinations
import torch
import logging
import cvxpy as cp
import sympy as sp
import numpy as np
from sympy import BlockDiagMatrix
from scipy.linalg import block_diag

from src.models.multi_layer import (identity_map,
                                    custom_net,
                                    MultiLayerNN)

from src.algorithms.abstract_verifier import _vector_for_separating_hyperplane
from src.algorithms.relaxations import (_relaxation_for_half_space,
                                        _build_Q_for_relu,
                                        _relaxation_for_hypercube)

from src.algorithms.certify_affine import CertifyAffine
from src.algorithms.certify_symbolic import (symbolic_build_T,
                                             symbolic_relaxation_for_half_space,
                                             symbolic_relaxation_for_relu,
                                             symbolic_relaxation_for_hypercube,
                                             symbolic_relaxation_for_polytope,
                                             symbolic_build_gamma)


class CertifyReLU(CertifyAffine):

    def __init__(self, f=None):
        super().__init__(f)
        self.name = 'CertifyReLU'


    def build_symbolic_matrices(self, x=None, eps=1):
        # for nn with final relu layer
        K = sum([w.shape[1] for w in self.nn_weights]) + self.nn_weights[-1].shape[0]
        x = np.array(x)
        dim_X = K + 1

        im_x = self.f(torch.tensor(x).T.float()).data.T.tolist()
        d = 0
        c = self.sep_hplane_for_advclass(x, complement=True)
        # logging.debug(c)

        P = symbolic_relaxation_for_hypercube(x=x, epsilon=eps)
        logging.info(f"P shape {P.shape} =")
        logging.info(sp.pretty(P))
        assert P.shape == (x.shape[0]+1, x.shape[0]+1)

        M_in_P = symbolic_build_M_in(P, self.nn_weights, K)
        logging.info(f"M_in_P shape {M_in_P.shape} =")
        logging.info(sp.pretty(M_in_P))
        assert M_in_P.shape == (dim_X, dim_X)

        dim = sum([w.shape[0] for w in self.nn_weights])
        Q, Q_vars = symbolic_relaxation_for_relu(dim=dim)
        logging.info(f"Q shape ({Q.shape}) = ")
        logging.info(sp.pretty(Q))
        assert Q.shape[0] == 2*dim + 1

        M_mid_Q = symbolic_build_M_mid(Q=Q, weights=self.nn_weights,
                                       bias_vecs=self.nn_bias_vecs, K=K)
        logging.info(f"M_mid_Q shape ({M_mid_Q.shape}) =")
        logging.info(sp.pretty(M_mid_Q))
        assert M_mid_Q.shape == (dim_X, dim_X)

        S, S_vals = symbolic_relaxation_for_half_space(c=c,
                                                       d=d,
                                                       dim_x=self.nn_weights[0].shape[1])
        logging.info(f"S in ({S.shape}) =")
        logging.info(sp.pretty(S.subs(S_vals)))
        assert S.shape == (self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1,
                           self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1)

        M_out_S, vals = symbolic_build_M_out(S, self.nn_weights,
                                             self.nn_bias_vecs, K)
        logging.info(f"M_out_S in ({M_out_S.shape}) =")
        logging.info(sp.pretty(S.subs(S_vals)))
        S_vals.update(vals)
        assert M_out_S.shape == (dim_X, dim_X)
        logging.info("M_out_S =")
        logging.info(sp.pretty(M_out_S.subs(S_vals)))

        # # print(M_in_P.shape)
        # # print(M_mid_Q.shape)
        assert M_in_P.shape == M_mid_Q.shape
        assert M_mid_Q.shape == M_out_S.shape
        # # X = M_in_P + M_mid_Q + M_out_S
        # # logging.info("X =")
        # # logging.info(sp.pretty(X))

    def network_constraints(self, x, eps, verbose=False):
        # for nn with final relu layer
        K = sum([w.shape[1] for w in self.nn_weights]) + self.nn_weights[-1].shape[0]
        x = np.array(x)
        dim_X = K + 1

        # input relaxation (with free vars)
        self.P, self.constraints, _ = _relaxation_for_hypercube(center=x,
                                                                epsilon=eps)
        assert self.P.shape == (x.shape[0]+1, x.shape[0]+1)

        M_in_P = build_M_in(self.P, self.nn_weights, self.nn_bias_vecs, K)
        assert M_in_P.shape == (dim_X, dim_X)

        # # relu relaxation (with free vars)
        dim = sum([w.shape[0] for w in self.nn_weights])
        self.Q, self.constraints = _build_Q_for_relu(dim=dim,
                                                     constraints=self.constraints)
        assert self.Q.shape[0] == 2*dim + 1

        M_mid_Q, self.constraints = build_M_mid(Q=self.Q,
                                                constraints=self.constraints,
                                                weights=self.nn_weights,
                                                bias_vecs=self.nn_bias_vecs, K=K)
        assert M_mid_Q.shape == (dim_X, dim_X)

        # output relaxation (fixed)
        d = 0
        c = self.sep_hplane_for_advclass(x, complement=True)
        self.S = _relaxation_for_half_space(c=c, d=d,
                                            dim_x=self.nn_weights[0].shape[1])
        assert self.S.shape == (self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1,
                                self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1)

        M_out_S = build_M_out(self.S, self.nn_weights, self.nn_bias_vecs, K)
        assert M_mid_Q.shape == (dim_X, dim_X)

        X = M_in_P + M_mid_Q + M_out_S
        self.constraints += [X << 0]
        return self.constraints

def _build_E(weights, i, K):
    assert (i == len(weights)-1) or (i == 0)
    _E = []
    if i == 0:
        _E.append(np.eye(weights[0].shape[1]))
        _E.append(np.zeros((weights[0].shape[1], K - weights[0].shape[1])))
        E = np.array(np.block([_E]))
    if i == len(weights)-1:
        _E.append(np.zeros((weights[-1].shape[1], K - weights[-1].shape[1])))
        _E.append(np.eye(weights[-1].shape[1]))
        E = np.array(np.block([_E]))
    return E



def symbolic_build_M_in(P, weights, K):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)
    assert(P.shape[0] == weights[0].shape[1] + 1)
    E0 = _build_E(weights, 0, K)
    assert E0.shape[1] == K
    E0 = sp.Matrix(E0)
    _in_ = sp.BlockMatrix([
        [E0,                       sp.zeros(E0.shape[0], 1)],
        [sp.zeros(1, E0.shape[1]),                sp.eye(1)],
    ])
    M_in_P = _in_.T @ P @ _in_
    return M_in_P


def symbolic_build_M_mid(Q, weights, bias_vecs, K):
    assert len(weights) == len(bias_vecs)

    A_weights = [sp.Matrix(w) for w in weights]
    A_rows = sum([w.shape[0] for w in A_weights])
    bias_concat = sp.Matrix(np.concatenate(bias_vecs))

    A = sp.BlockMatrix([
        [BlockDiagMatrix(*A_weights),
         sp.zeros(A_rows, weights[-1].shape[1])]
    ])
    # print(f"A in {A.shape}")
    # sp.pprint(A)
    assert A.shape[0] == bias_concat.shape[0]

    B_d_mat = BlockDiagMatrix(*[sp.eye(w.shape[1]) for w in weights[1:]], sp.eye(weights[-1].shape[0]))
    B_rows = B_d_mat.shape[0]
    B = sp.BlockMatrix([
        [sp.zeros(B_rows, weights[0].shape[1]), B_d_mat]
    ])
    # print(f"B in {B.shape}")
    # sp.pprint(B)
    assert B.shape[0] == bias_concat.shape[0]
    assert B.shape[1] == A.shape[1]

    _mid_ = sp.BlockMatrix([
        [A,                        sp.Matrix(bias_concat)],
        [B,                       sp.zeros(B.shape[0], 1)],
        [sp.zeros(1, B.shape[1]),               sp.eye(1)]
    ])
    # print(f"_mid_ in {_mid_.shape}")
    # sp.pprint(_mid_)
    # this assertion maybe not universal
    assert _mid_.shape[0] == Q.shape[1]

    M_mid_Q = _mid_.T @ Q @ _mid_
    return M_mid_Q


def symbolic_build_M_out(S, weights, bias_vecs, K):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = sp.Matrix(_build_E(weights, 0, K))
    El = sp.Matrix(_build_E(weights, len(weights)-1, K))
    assert E0.shape[1] == El.shape[1]

    _E0 = sp.MatrixSymbol('E0', E0.shape[0], E0.shape[1])
    _El = sp.MatrixSymbol('El', El.shape[0], El.shape[1])
    _S = sp.MatrixSymbol('S', S.shape[0], S.shape[1])

    bv = sp.Matrix(np.array([bias_vecs[-1]]).T)
    _out_ = sp.BlockMatrix([
        [_E0,                           sp.zeros(E0.shape[0], 1)],
        [_El,                           sp.zeros(El.shape[0], 1)],
        [sp.zeros(1, E0.shape[1]),                     sp.eye(1)]
    ])
    M_out = sp.MatMul(_out_.T @ _S @ _out_)
    vals = {_E0: E0, _El: El, _S: S}
    return M_out, vals



def build_M_in(P, weights, bias_vecs, K):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)
    assert(P.shape[0] == weights[0].shape[1] + 1)
    E0 = _build_E(weights, 0, K)
    _in_ = cp.bmat([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [np.zeros((1, E0.shape[1])),                  np.eye(1)],
    ])
    M_in_P = _in_.T @ P @ _in_
    return M_in_P


def build_M_mid(Q, constraints, weights, bias_vecs, K, activ_func='relu'):
    assert len(weights) == len(bias_vecs)

    A_weights = [np.array(w) for w in weights]
    A_rows = sum([w.shape[0] for w in A_weights])
    bias_concat = np.reshape(np.concatenate(bias_vecs), (-1, 1))

    A = np.bmat([
        [block_diag(*A_weights),
         sp.zeros(A_rows, weights[-1].shape[1])]
    ])
    assert A.shape[0] == bias_concat.shape[0]

    B_d_mat = block_diag(*[np.eye(w.shape[1]) for w in weights[1:]],
                         np.eye(weights[-1].shape[0]))
    B_rows = B_d_mat.shape[0]
    B = np.bmat([
        [np.zeros((B_rows, weights[0].shape[1])), B_d_mat]
    ])
    assert B.shape == A.shape

    _mid_ = np.bmat([
        [A,                             np.array(bias_concat)],
        [B,                         np.zeros((B.shape[0], 1))],
        [np.zeros((1, B.shape[1])),                 np.eye(1)]
    ])
    # this assertion maybe not universal to every problem?
    assert _mid_.shape[0] == Q.shape[1]

    M_mid_Q = _mid_.T @ Q @ _mid_
    return M_mid_Q, constraints


def build_M_out(S, weights, bias_vecs, K):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = _build_E(weights, 0, K)
    El = _build_E(weights, len(weights)-1, K)
    assert E0.shape[1] == El.shape[1]
    _out_ = np.block([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [El,                         np.zeros((El.shape[0], 1))],
        [np.zeros((1, E0.shape[1])),                  np.eye(1)]
    ])
    return _out_.T @ S @ _out_


def quick_test_eps_robustness():
    f = custom_net()
    x = [[4], [2]]
    cert = CertifyReLU(f)
    eps = 1
    e_robust = cert.decide_eps_robustness(x, eps, verbose=False)

    x_class = f.class_for_input(x)
    print(f"f({x}) = class {x_class+1}")
    print(f"{f.name} is ({eps})-robust at {x}?  {e_robust}")
    print("P = ")
    print(cert.P.value)


def symbolic_test():
    f = custom_net()
    cert = CertifyReLU(f)
    eps = 1
    x = [[4], [2]]
    cert.build_symbolic_matrices(x, eps)


if __name__ == '__main__':
    # quick_test_eps_robustness()
    symbolic_test()
