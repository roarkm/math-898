from itertools import combinations
import cvxpy as cp
import numpy as np


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


def _relaxation_for_half_space(c, d, dim_x, tol=10**(-3)):
    # for half space defined by {y : cy < d} (in the output space of NN)
    # dim_x is the input space dimension of the NN
    dim_c = c.shape[0]  # output dimension of NN
    S = np.block([
        [np.zeros((dim_x, dim_x)), np.zeros((dim_x, dim_c)), np.zeros((dim_x, 1))],
        [np.zeros((dim_c, dim_x)), np.zeros((dim_c, dim_c)),                    c],
        [np.zeros((1, dim_x)),     c.T,                  -2*np.array([[d]]) + tol]
    ])
    return S


def _build_T(dim, constraints=[]):
    T = cp.bmat(np.zeros((dim, dim)))
    index_combos = list(combinations(range(0, dim), 2))
    lambdas_ij = cp.Variable(len(index_combos), name='lambda_ij')
    _constraints = [lambdas_ij >= 0]
    for k, (i, j) in enumerate(index_combos):
        ei = np.zeros((dim, 1))
        ei[i] = 1
        ej = np.zeros((dim, 1))
        ej[j] = 1
        v = cp.bmat([ei - ej])
        T += lambdas_ij[k] * cp.matmul(v.T, v)
    return T, _constraints


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
