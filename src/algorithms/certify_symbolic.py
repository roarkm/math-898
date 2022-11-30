from itertools import combinations
import numpy as np
import sympy as sp
from sympy import BlockDiagMatrix
from src.models.multi_layer import (MultiLayerNN,
                                    identity_map)


def _build_E(weights, i):
    assert (i == len(weights)-1) or (i == 0)
    _E = []
    if i == 0:
        _E.append(np.eye(weights[0].shape[1]))
        _E.append(np.zeros((weights[0].shape[1],
                            sum([w.shape[1] for w in weights[1:]]))))
        E = np.array(np.block([_E]))
    if i == len(weights)-1:
        _E.append(np.zeros((weights[-1].shape[1],
                            sum([w.shape[1] for w in weights[:-1]]))))
        _E.append(np.eye(weights[-1].shape[1]))
        E = np.array(np.block([_E]))
    assert E.shape[1] == sum([w.shape[1] for w in weights])
    return E

def symbolic_build_M_out(S, weights, bias_vecs):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.
    E0 = sp.Matrix(_build_E(weights[:-1], 0))
    El = sp.Matrix(_build_E(weights, len(weights)-1))
    assert E0.shape[1] == El.shape[1]

    _E0 = sp.MatrixSymbol('E0', E0.shape[0], E0.shape[1])
    _El = sp.MatrixSymbol('El', El.shape[0], El.shape[1])
    _S = sp.MatrixSymbol('S', S.shape[0], S.shape[1])
    __bv = np.array([bias_vecs[-1]]).T
    _bv = sp.MatrixSymbol('bv', __bv.shape[0], __bv.shape[1])

    bv = sp.Matrix(np.array([bias_vecs[-1]]).T)
    _out_ = sp.BlockMatrix([
        [_E0,                           sp.zeros(E0.shape[0], 1)],
        [sp.Matrix(weights[-1]) @ _El,                       _bv],
        [sp.zeros(1, E0.shape[1]),                     sp.eye(1)]
    ])
    M_out = sp.MatMul(_out_.T @ _S @ _out_)
    vals = {_E0: E0, _El: El, _S: S, _bv: bv}
    return M_out, vals


def symbolic_build_M_in(P, weights, bias_vecs):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)
    assert(P.shape[0] == weights[0].shape[1] + 1)
    E0 = _build_E(weights[:-1], 0)
    # print(f"E0 shape {E0.shape}")
    E0 = sp.Matrix(E0)
    _in_ = sp.BlockMatrix([
        [E0,                       sp.zeros(E0.shape[0], 1)],
        [sp.zeros(1, E0.shape[1]),                sp.eye(1)],
    ])
    M_in_P = _in_.T @ P @ _in_
    return M_in_P


def symbolic_build_M_mid(Q, weights, bias_vecs):
    assert len(weights) == len(bias_vecs)

    A_weights = [sp.Matrix(w) for w in weights[:-1]]
    A_rows = sum([w.shape[0] for w in A_weights])
    bias_concat = sp.Matrix(np.concatenate(bias_vecs[:-1]))

    A = sp.BlockMatrix([
        [BlockDiagMatrix(*A_weights),
         sp.zeros(A_rows, weights[-1].shape[1])]
    ])
    # print(f"A in {A.shape}")
    # sp.pprint(A)
    assert A.shape[0] == bias_concat.shape[0]

    B_d_mat = BlockDiagMatrix(*[sp.eye(w.shape[1]) for w in weights[1:]])
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
    zetas = [sp.symbols(f"zeta{i}") for i in range(0, dim)]
    nus = sp.Matrix([[sp.symbols(f"nu{i}")] for i in range(0, dim)])
    etas = sp.Matrix([[sp.symbols(f"eta{i}")] for i in range(0, dim)])

    # keep track of free sympy variables
    Q_vars = Q_vars + [nu for nu in nus]
    Q_vars = Q_vars + zetas
    Q_vars = Q_vars + [nu for nu in nus]

    diag_lambda = sp.diag(*zetas)

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
    return Q, Q_vars


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


def symbolic_build_gamma(dim):
    # build a symmetric  martix G,
    # with G_ii == 0 (zeros on the diag)
    # how to ensure PSD? (easy to do with cvxpy, sympy though?)
    free_vars = {}
    for i in range(1, dim):
        free_vars[dim-i] = [sp.symbols(f"g{i}{j}") for j in range(1,i+1)]
        free_vars[-(dim-i)] = [sp.symbols(f"g{i}{j}") for j in range(1,i+1)]
    return sp.matrices.sparsetools.banded(free_vars)


def symbolic_relaxation_for_polytope(H, b):
    # symbolic relaxation for polytope { x | Hx <= b}
    assert H.shape[0] == b.shape[0]
    Gamma = symbolic_build_gamma(H.shape[0])
    P = sp.BlockMatrix([
        [H.T @ Gamma @ H,       -1 * H.T @ Gamma @ b],
        [-1 * b.T @ Gamma @ H,       b.T @ Gamma @ b],
    ])
    return P


def symbolic_relaxation_for_half_space(c, d, dim_x):
    # for half space defined by {y : cy < d} (in the output space of NN)
    # dim_x is the input space dimension of the NN
    dim_c = c.shape[0]
    _c = sp.MatrixSymbol('c', c.shape[0], c.shape[1])
    _d = sp.MatrixSymbol('d', 1, 1)

    # temp turn of warnings
    # sympy wants np.array instead of sp.Matrix but
    # then subbing in values does not work
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    vals = {_c: sp.Matrix([c]).T, _d: sp.Matrix([[d]])}
    # this does not warn (but introduces a substituion bug at S.subs()
    # vals = {_c: np.array([c]).T, _d: np.array([[d]])}
    warnings.filterwarnings("default", category=DeprecationWarning)

    S = sp.BlockMatrix([
        [sp.zeros(dim_x, dim_x), sp.zeros(dim_x, dim_c),    sp.zeros(dim_x,1)],
        [sp.zeros(dim_c, dim_x), sp.zeros(dim_c, dim_c),                   _c],
        [sp.zeros(1, dim_x),     _c.T,                                  -2*_d]
    ])
    return S, vals

if __name__ == '__main__':
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
    # dim = (sum([w.shape[0] for w in weights[:-1]])
         # + sum([w.shape[1] for w in weights[-1:]]))
    print(_build_E(weights,0).shape)
    print(_build_E(weights,2).shape)
    # exit()

    dim = sum([w.shape[0] for w in weights[:-1]])
    # print(dim)
    Q, Q_vars = symbolic_relaxation_for_relu(dim=dim)
    assert Q.shape[0] == 2*dim + 1
    print(f"Q shape {Q.shape}")
    symbolic_build_M_mid(Q=Q, weights=weights, bias_vecs=b)
    exit()
