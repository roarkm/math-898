from itertools import combinations
import torch
import sys
import logging
import cvxpy as cp
import sympy as sp
import numpy as np
from scipy.linalg import block_diag

from src.models.multi_layer import (identity_map,
                                    MultiLayerNN)

from src.algorithms.abstract_verifier import _vector_for_separating_hyperplane
from src.algorithms.relaxations import (_relaxation_for_half_space,
                                        _build_Q_for_relu,
                                        _relaxation_for_hypercube)
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


class CertifyAffine():

    def __init__(self, f=None):
        self.f = f
        self.name = 'CertifyAffine'
        self.solver = cp.CVXOPT
        self.counter_example = None
        # self.solver = cp.SCS
        self.max_iters = 10**16
        if f:
            self.nn_weights, self.nn_bias_vecs = f.get_weights()
        # logging.basicConfig(format='Certify-%(levelname)s:\n%(message)s',
                            # level=logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('\n')
        # handler.setFormatter(formatter)
        root.addHandler(handler)

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
        x = np.array(x)
        n = sum([w.shape[0] for w in self.nn_weights[:-1]])
        dim_X = n + self.nn_weights[0].shape[1] + 1

        # input relaxation (with free vars)
        self.P, self.constraints, _ = _relaxation_for_hypercube(center=x,
                                                                epsilon=eps)
        assert self.P.shape == (x.shape[0]+1, x.shape[0]+1)

        M_in_P = build_M_in(self.P, self.nn_weights, self.nn_bias_vecs)
        assert M_in_P.shape == (dim_X, dim_X)

        # relu relaxation (with free vars)
        dim = sum([w.shape[0] for w in self.nn_weights[:-1]])
        self.Q, self.constraints = _build_Q_for_relu(dim=dim,
                                                     constraints=self.constraints)
        assert self.Q.shape[0] == 2*dim + 1

        M_mid_Q, self.constraints = build_M_mid(Q=self.Q,
                                                constraints=self.constraints,
                                                weights=self.nn_weights,
                                                bias_vecs=self.nn_bias_vecs)
        assert M_mid_Q.shape == (dim_X, dim_X)

        # output relaxation (fixed)
        d = 0
        c = self.sep_hplane_for_advclass(x, complement=True)
        self.S = _relaxation_for_half_space(c=c, d=d,
                                            dim_x=self.nn_weights[0].shape[1])
        assert self.S.shape == (self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1,
                           self.nn_weights[0].shape[1] + self.nn_weights[-1].shape[0] + 1)

        M_out_S = build_M_out(self.S, self.nn_weights, self.nn_bias_vecs)
        assert M_mid_Q.shape == (dim_X, dim_X)

        X = M_in_P + M_mid_Q + M_out_S
        self.constraints += [X << 0]
        return self.constraints

    def str_constraints(self):
        s = ''
        for c in self.constraints:
            s += f"Constraint {c.constr_id} -----------------------------\n"
            s += str(c)
            s += "\n"
        return s

    def assert_valid_ref_point(self, x):
        fx = self.f.forward(torch.tensor(x).T.float()).detach().numpy().T
        sorted_img = np.sort(fx)
        assert sorted_img[-1] != sorted_img[-2], \
               f"Ambiguous reference input: {x} |--> {fx.T}"

    def _build_eps_robustness_problem(self, x=[[9], [0]], eps=1, verbose=False):
        self.assert_valid_ref_point(x)
        self.network_constraints(x=x, eps=eps, verbose=verbose)
        self.prob = cp.Problem(cp.Minimize(1), self.constraints)

    def _decide_eps_robustness(self, verbose=False, max_iters=10**6):
        self.prob.solve(verbose=verbose,
                        max_iter=self.max_iters,
                        solver=self.solver)
        status = self.prob.status

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

    def decide_eps_robustness(self, x=[[9], [0]], eps=1,
                              verbose=False, max_iters=10**6):
        self._build_eps_robustness_problem(x, eps, verbose)
        return self._decide_eps_robustness(verbose, max_iters)

    def compute_robustness(self, x, verbose=False):
        raise Exception("compute_robustness() not supported by Certify.")


def build_M_out(S, weights, bias_vecs):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = _build_E(weights, 0)
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
    E0 = _build_E(weights, 0)
    _in_ = cp.bmat([
        [E0,                         np.zeros((E0.shape[0], 1))],
        [np.zeros((1, E0.shape[1])),                  np.eye(1)],
    ])
    M_in_P = _in_.T @ P @ _in_
    return M_in_P


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


def quick_test_eps_robustness():
    f = identity_map(2, 2)
    x = [[9], [1]]
    cert = CertifyAffine(f)
    eps = 1
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
    cert = CertifyAffine(f)
    eps = 8
    x = [[9], [1], [1]]
    cert.build_symbolic_matrices(x, eps)


if __name__ == '__main__':
    quick_test_eps_robustness()
    # symbolic_test()

