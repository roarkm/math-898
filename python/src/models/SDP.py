import numpy as np
import cvxpy as cp
from itertools import combinations

def build_M_out(S, W0, b0, W1, b1):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f.
    # There are no free variables in the returned matrix.

    W0_in_dim =  W0.shape[1]
    W0_out_dim = W0.shape[0]
    W1_in_dim =  W1.shape[1]
    W1_out_dim = W1.shape[0]
    assert(W0_out_dim == W1_in_dim)
    assert(W0_out_dim == b0.shape[0])
    assert(W1_out_dim == b1.shape[0])
    assert(S.shape[0] == W0_in_dim + W1_out_dim + 1)
    assert(S.shape[1] == W0_in_dim + W1_out_dim + 1)

    _out_ = np.block([
        [np.eye(W0_in_dim),                 np.zeros((W0_in_dim, W1_in_dim)), np.zeros((W0_in_dim, 1))],
        [np.zeros((W1_out_dim, W0_in_dim)), W1,                               b1],
        [np.zeros((1, W0_in_dim)),          np.zeros((1, W1_in_dim)),         np.eye(1)]
    ])
    return _out_.T * S * _out_


def build_M_in(P, W0, b0, W1, b1):
    # P is specified by the caller and quadratically overapproximates
    # the region of interest in the input space of f (typically a hypercube)

    W0_in_dim =  W0.shape[1]
    W0_out_dim = W0.shape[0]
    W1_in_dim =  W1.shape[1]
    W1_out_dim = W1.shape[0]
    assert(W0_out_dim == W1_in_dim)
    assert(W0_out_dim == b0.shape[0])
    assert(W1_out_dim == b1.shape[0])
    assert(P.shape[0] == W0_in_dim + 1)

    _in_ = cp.bmat([
        [np.eye(W0_in_dim),        np.zeros((W0_in_dim, W1_out_dim)), np.zeros((W0_in_dim, 1))],
        [np.zeros((1, W0_in_dim)), np.zeros((1, W1_out_dim)),         np.eye(1)]
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

def build_M_mid(Q, constraints, W0, b0, W1, b1, activ_func='relu'):
    W0_in_dim =  W0.shape[1]
    W0_out_dim = W0.shape[0]
    W1_in_dim =  W1.shape[1]
    W1_out_dim = W1.shape[0]
    assert(W0_out_dim == W1_in_dim)
    assert(W0_out_dim == b0.shape[0])
    assert(W1_out_dim == b1.shape[0])
    assert(Q.shape[0] == W0_in_dim + W1_out_dim + 1)
    assert(Q.shape[0] == Q.shape[1])

    _mid_ = cp.bmat([
        [W0, np.zeros((W0_out_dim, W1_in_dim)), b0],
        [np.zeros((W1_out_dim, W0_out_dim)), np.eye(W1_in_dim), np.zeros((W1_out_dim, 1))],
        [np.zeros((1, W0_in_dim)), np.zeros((1, W1_in_dim)), np.eye(1)]
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
        [np.zeros((dim_x, dim_x)),     np.zeros((dim_x, dim_c)),     np.zeros((dim_x,1))],
        [np.zeros((dim_c, dim_x)), np.zeros((dim_c, dim_c)), c],
        [np.zeros((1, dim_x)),     c.T,            -2*np.matrix([d])]
    ])
    return S

def two_by_two_test():
    x = np.matrix('9; 1')
    eps=1.005

    W0 = np.matrix('1 0; 0 1')
    b0 = np.matrix('0; 0')
    W1 = np.matrix('.99 0; 0 .99')
    b1 = np.matrix('1; 1')

    im_x = W1 @ np.maximum(0, W0 @ x + b0) + b1
    klass = np.argmax(im_x)
    print(f"f({x.T}) = {im_x.T} |--> class: {klass}")

    # TODO: make this work for higher dimensions
    d=0
    c = np.matrix('-1; 1') # defines halfspace where y1 > y2
    if klass == 1:
        c = -1 * c # defines halfspace where y2 > y1

    P, constraints = _relaxation_for_hypercube(x=x, epsilon=eps)
    Q, constraints = _build_Q_for_relu(dim=W0.shape[0], constraints=constraints)
    S = _relaxation_for_half_space(c=c, d=d, dim_x=W0.shape[1])

    M_in_P = build_M_in(P=P, W0=W0, b0=b0, W1=W1, b1=b1)
    M_mid_Q, constraints = build_M_mid(Q=Q, constraints=constraints,
                                       W0=W0, b0=b0, W1=W1, b1=b1)
    M_out_S = build_M_out(S=S, W0=W0, b0=b0, W1=W1, b1=b1)

    X = M_in_P + M_mid_Q + M_out_S
    constraints += [X << 0]

    prob = cp.Problem(cp.Minimize(1), constraints)
    prob.solve()

    status = prob.status
    if status == cp.OPTIMAL:
        print(f"Certificate that all x within {eps} inf-norm of {x.T} are classified as class {klass}")
    elif status == cp.INFEASIBLE:
        print(f"Could NOT verify that all x within {eps} inf-norm of {x.T} are classified as {klass}")
    # TODO:
    # elif status == cp.UNBOUNDED:
    # elif status == cp.OPTIMAL_INACCURATE:
    # elif status == cp.INFEASIBLE_INACCURATE:
    # elif status == cp.UNBOUNDED_INACCURATE:
    # elif status == cp.INFEASIBLE_OR_UNBOUNDED:


if __name__ == '__main__':
    two_by_two_test()
