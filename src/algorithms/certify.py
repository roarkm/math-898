from itertools import combinations
import torch
import logging
import cvxpy as cp
import sympy as sp
import numpy as np
from scipy.linalg import block_diag

from src.models.multi_layer import (identity_map,
                                    MultiLayerNN)

from src.algorithms.abstract_verifier import _vector_for_separating_hyperplane

from src.algorithms.certify_symbolic import (symbolic_build_M_out,
                                             symbolic_build_M_in,
                                             symbolic_build_M_mid,
                                             symbolic_build_T,
                                             symbolic_relaxation_for_half_space,
                                             symbolic_relaxation_for_relu,
                                             symbolic_relaxation_for_hypercube,
                                             symbolic_relaxation_for_polytope,
                                             symbolic_build_gamma,
                                             _build_E)


class Certify():

    def __init__(self, f=None):
        self.f = f
        self.name = 'Certify'
        if f:
            self.nn_weights, self.nn_bias_vecs = f.get_weights()
        logging.basicConfig(format='Certify-%(levelname)s:\n%(message)s',
                            level=logging.DEBUG)

    def sep_hplane_for_advclass(self, x, complement=True):
        out_shape = self.f(torch.tensor(x).T.float()).shape[1]
        x_class, adversarial_class = self.f.top_two_classes(x)
        return _vector_for_separating_hyperplane(large_index=x_class,
                                                 small_index=adversarial_class,
                                                 n=out_shape,
                                                 complement=complement)

    def build_symbolic_matrices(self, x=None, eps=1):
        # for nn with final affine layer
        n = sum([w.shape[0] for w in self.nn_weights[:-1]])
        x = np.array(x)
        dim_X = n + self.nn_weights[0].shape[1] + 1
        print(f"dim_X = {dim_X}")

        # im_x = self.f(torch.tensor(x).T.float()).data.T.tolist()
        d = 0
        c = self.sep_hplane_for_advclass(x, complement=True)
        # logging.debug(c)

        P = symbolic_relaxation_for_hypercube(x=x, epsilon=eps)
        logging.info(f"P shape {P.shape} =")
        # logging.info(sp.pretty(P))
        assert P.shape == (x.shape[0]+1, x.shape[0]+1)

        M_in_P = symbolic_build_M_in(P, self.nn_weights, self.nn_bias_vecs)
        logging.info(f"M_in_P shape {M_in_P.shape} =")
        # logging.info(sp.pretty(M_in_P))
        assert M_in_P.shape == (dim_X, dim_X)

        dim = sum([w.shape[0] for w in self.nn_weights[:-1]])
        Q, Q_vars = symbolic_relaxation_for_relu(dim=dim)
        logging.info(f"Q shape ({Q.shape}) = ")
        # logging.info(sp.pretty(Q))
        assert Q.shape[0] == 2*dim + 1

        M_mid_Q = symbolic_build_M_mid(Q=Q, weights=self.nn_weights,
                                       bias_vecs=self.nn_bias_vecs)
        logging.info(f"M_mid_Q shape ({M_mid_Q.shape}) =")
        # logging.info(sp.pretty(M_mid_Q))
        assert M_mid_Q.shape == (dim_X, dim_X)

        S, S_vals = symbolic_relaxation_for_half_space(c=c,
                                                       d=d,
                                                       dim_x=self.nn_weights[0].shape[1])
        logging.info(f"S in ({S.shape}) =")
        # logging.info(sp.pretty(S.subs(S_vals)))
        assert S.shape == (self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1,
                           self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1)

        M_out_S, vals = symbolic_build_M_out(S, self.nn_weights,
                                             self.nn_bias_vecs)
        logging.info(f"M_out_S in ({M_out_S.shape}) =")
        # logging.info(sp.pretty(S.subs(S_vals)))
        S_vals.update(vals)
        assert M_out_S.shape == (dim_X, dim_X)
        # logging.info("M_out_S =")
        # logging.info(sp.pretty(M_out_S.subs(S_vals)))

        # print(M_in_P.shape)
        # print(M_mid_Q.shape)
        assert M_in_P.shape == M_mid_Q.shape
        assert M_mid_Q.shape == M_out_S.shape
        # X = M_in_P + M_mid_Q + M_out_S
        # logging.info("X =")
        # logging.info(sp.pretty(X))

    def network_constraints(self, x, eps, verbose=False):
        # for nn with final affine layer
        n = sum([w.shape[0] for w in self.nn_weights[:-1]])  # num neurons in f
        x = np.array(x)
        dim_X = n + self.nn_weights[0].shape[1] + 1

        d = 0
        c = self.sep_hplane_for_advclass(x, complement=True)
        self.P, self.constraints, _ = _relaxation_for_hypercube(center=x,
                                                                epsilon=eps)
        assert self.P.shape == (x.shape[0]+1, x.shape[0]+1)

        M_in_P = build_M_in(self.P, self.nn_weights, self.nn_bias_vecs)
        assert M_in_P.shape == (dim_X, dim_X)

        dim = sum([w.shape[0] for w in self.nn_weights[:-1]])
        self.Q, self.constraints = _build_Q_for_relu(dim=dim,
                                                     constraints=self.constraints)
        assert self.Q.shape[0] == 2*dim + 1
        M_mid_Q, self.constraints = build_M_mid(Q=self.Q,
                                                constraints=self.constraints,
                                                weights=self.nn_weights,
                                                bias_vecs=self.nn_bias_vecs)
        assert M_mid_Q.shape == (dim_X, dim_X)

        self.S = _relaxation_for_half_space(c=c, d=d,
                                            dim_x=self.nn_weights[0].shape[1])
        assert self.S.shape == (self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1,
                           self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1)
        M_out_S = build_M_out(self.S, self.nn_weights, self.nn_bias_vecs)
        assert M_mid_Q.shape == (dim_X, dim_X)

        X = M_in_P + M_mid_Q + M_out_S
        self.constraints += [X << 0]
        return self.constraints

    def decide_eps_robustness(self, x=[[9], [0]], eps=1,
                              verbose=False, max_iters=10**6):

        self.network_constraints(x=x, eps=eps, verbose=verbose)
        prob = cp.Problem(cp.Minimize(1), self.constraints)
        prob.solve(verbose=verbose,
                   max_iters=max_iters,
                   solver=cp.CVXOPT)
        status = prob.status

        if (status == cp.OPTIMAL_INACCURATE) or \
           (status == cp.INFEASIBLE_INACCURATE) or \
           (status == cp.UNBOUNDED_INACCURATE):
            logging.warning("Warning: inaccurate solution.")

        if (status == cp.OPTIMAL) or \
           (status == cp.OPTIMAL_INACCURATE):
            # TODO: verify 0-level set of P contains region of interest
            # (test corner points)
            self.P = self.P.value
            self.Q = self.Q.value
            return True
        elif (status == cp.INFEASIBLE) or \
             (status == cp.INFEASIBLE_INACCURATE):
            return False
        else:
            raise Exception(status)

    def compute_robustness(self, x, verbose=False):
        raise Exception("compute_robustness() not supported by Certify.")


def build_M_out(S, weights, bias_vecs):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = _build_E(weights[:-1], 0)
    El = _build_E(weights, len(weights)-1)
    assert E0.shape[1] == El.shape[1]
    _out_ = np.block([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [weights[-1]@El,            np.array([bias_vecs[-1]]).T],
        [np.zeros((1, E0.shape[1])),                  np.eye(1)]
    ])
    return _out_.T @ S @ _out_


def build_M_in(P, weights, bias_vecs):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)
    assert(P.shape[0] == weights[0].shape[1] + 1)
    E0 = _build_E(weights[:-1], 0)
    _in_ = cp.bmat([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [np.zeros((1, E0.shape[1])),                  np.eye(1)],
    ])
    M_in_P = _in_.T @ P @ _in_
    return M_in_P


def _build_T(dim, constraints=[]):
    T = cp.bmat(np.zeros((dim, dim)))
    index_combos = list(combinations(range(0, dim), 2))
    lambdas_ij = cp.Variable(len(index_combos), name='lambda_ij')
    constraints += [lambdas_ij >= 0]
    for k, (i, j) in enumerate(index_combos):
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
    Q33 = np.zeros((1, 1))

    Q = cp.bmat([
        [Q11,   Q12,   Q13],
        [Q12.T, Q22,   Q23],
        [Q13.T, Q23.T, Q33],
    ])
    return Q, constraints


def build_M_mid(Q, constraints, weights, bias_vecs, activ_func='relu'):
    assert len(weights) == len(bias_vecs)

    A_weights = [np.array(w) for w in weights[:-1]]
    A_rows = sum([w.shape[0] for w in A_weights])
    bias_concat = np.reshape(np.concatenate(bias_vecs[:-1]), (-1, 1))

    A = np.bmat([
        [block_diag(*A_weights),
         sp.zeros(A_rows, weights[-1].shape[1])]
    ])
    assert A.shape[0] == bias_concat.shape[0]

    B_d_mat = block_diag(*[np.eye(w.shape[1]) for w in weights[1:]])
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


def _relaxation_for_hypercube(center, epsilon, constraints=[], values=None):
    n = center.shape[0]
    x_floor = center - epsilon * np.ones((n, 1))
    x_ceil = center + epsilon * np.ones((n, 1))

    if values is None:
        a = cp.Variable(n, name='a')
        Gamma = cp.diag(a)
        constraints += [a >= 0]
    else:
        assert values.all() >= 0
        Gamma = np.diag(values)
        assert Gamma.shape[0] == n
        a = None

    P = cp.bmat([
        [-2 * Gamma,                        Gamma @ (x_floor + x_ceil)],
        [(x_floor + x_ceil).T @ Gamma, -2 * x_floor.T @ Gamma @ x_ceil]
    ])
    return P, constraints, a


def relaxation_for_polytope(H, b):
    # relaxation for polytope { x | Hx <= b}
    Gamma = cp.Variable((H.shape[0], H.shape[0]), PSD=True)
    constr_eq = cp.diag(Gamma) == 0
    P = np.block([
        [H.T @ Gamma @ H,       -1 * H.T @ Gamma @ b],
        [-1 * b.T @ Gamma @ H,       b.T @ Gamma @ b],
    ])
    return P, constr_eq


def _relaxation_for_half_space(c, d, dim_x):
    # for half space defined by {y : cy < d} (in the output space of NN)
    # dim_x is the input space dimension of the NN
    dim_c = c.shape[0]
    S = np.block([
        [np.zeros((dim_x, dim_x)), np.zeros((dim_x, dim_c)), np.zeros((dim_x, 1))],
        [np.zeros((dim_c, dim_x)), np.zeros((dim_c, dim_c)),                    c],
        [np.zeros((1, dim_x)),     c.T,                        -2*np.array([[d]])]
    ])
    return S


def quick_test_eps_robustness():
    # f = identity_map(2, 2)
    b = [
        np.array([[1],
                  [1]]),
        np.array([[2],
                  [2],
                  [2]]),
        np.array([[3],
                  [3]]),
    ]
    weights = [
        np.array([[1, 1, 1],
                  [1, 1, 1]]),
        np.array([[2, 2],
                  [2, 2],
                  [2, 2]]),
        np.array([[3, 3, 3],
                  [3, 3, 3]]),
    ]
    f = MultiLayerNN(weights, b)
    cert = Certify(f)
    eps = 1
    x = [[5], [1], [1]]
    e_robust = cert.decide_eps_robustness(x, eps, verbose=False)

    x_class = f.class_for_input(x)
    print(f"f({x}) = class {x_class+1}")
    print(f"{f.name} is ({eps})-robust at {x}?  {e_robust}")


def symbolic_test():
    b = [
        np.array([[1],
                  [1]]),
        np.array([[2],
                  [2],
                  [2]]),
        np.array([[3],
                  [3]]),
    ]
    weights = [
        np.array([[1, 1, 1],
                  [1, 1, 1]]),
        np.array([[2, 2],
                  [2, 2],
                  [2, 2]]),
        np.array([[3, 3, 3],
                  [3, 3, 3]]),
    ]
    f = MultiLayerNN(weights, b)
    # f = identity_map(2, 2)
    cert = Certify(f)
    eps = 8
    x = [[9], [1], [1]]
    cert.build_symbolic_matrices(x, eps)


if __name__ == '__main__':
    quick_test_eps_robustness()
    # symbolic_test()
