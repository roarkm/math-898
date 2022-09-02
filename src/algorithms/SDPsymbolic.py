import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cvxpy as cp
import sympy as sp
from itertools import combinations


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

def verify_toy_nn_at_point(nn, reference_point, epsilon):
    # expects a list of weight matrices,
    # list of bias vectors,
    # reference point in the input space,
    # and an epsilon for infinity ball radius

    return None


def test():
    # Generate a random SDP.
    n = 3
    p = 3
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n,n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                      constraints)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)

def build_M_out(S, W0, b0, W1, b1, logging=False):
    # S is specified by caller and quadratically overapproximates
    # the safety set in the graph of f
    # TODO: build helper to extract weights from a pytorch NN

    W0_in_dim =  W0.shape[1]
    W0_out_dim = W0.shape[0]
    W1_in_dim =  W1.shape[1]
    W1_out_dim = W1.shape[0]
    assert(W0_out_dim == W1_in_dim)
    assert(W0_out_dim == b0.shape[0])
    assert(W1_out_dim == b1.shape[0])
    assert(S.shape[0] == W0_in_dim + W1_out_dim + 1)
    assert(S.shape[1] == W0_in_dim + W1_out_dim + 1)
    assert(S.is_symmetric())

    _W0 = sp.MatrixSymbol('W0', W0_out_dim, W0_in_dim)
    _b0 = sp.MatrixSymbol('b0', W0_out_dim, 1)
    _W1 = sp.MatrixSymbol('W1', W1_out_dim, W1_in_dim)
    _b1 = sp.MatrixSymbol('b1', W1_out_dim, 1)
    _S =  sp.MatrixSymbol('S',  W0_in_dim + W1_in_dim + 1, W0_in_dim + W1_in_dim + 1)
    vals={_W1: W1, _b1: b1, _S: S}

    _out_ = sp.BlockMatrix([
            [sp.eye(W0_in_dim),               sp.zeros(W0_in_dim, W1_in_dim), sp.zeros(W0_in_dim, 1)],
            [sp.zeros(W1_out_dim, W0_in_dim), _W1,                            _b1],
            [sp.zeros(1, W0_in_dim),          sp.zeros(1, W1_in_dim),         sp.eye(1)]
         ])

    M_out_S = sp.MatMul(_out_.T, _S, _out_)

    assert(sp.Matrix(M_out_S).subs(vals).is_symmetric())

    if logging:
        sp.pprint(M_out_S)
        sp.pprint(M_out_S.as_explicit())
        sp.pprint(M_out_S.subs(vals).as_explicit())

    return sp.Matrix(M_out_S.subs(vals))

def build_M_in(P, P_vals, W0, b0, W1, b1, logging=False):
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
    assert(P.is_symmetric())

    _W0 = sp.MatrixSymbol('W0', W0_out_dim, W0_in_dim)
    _b0 = sp.MatrixSymbol('b0', W0_out_dim, 1)
    _W1 = sp.MatrixSymbol('W1', W1_out_dim, W1_in_dim)
    _b1 = sp.MatrixSymbol('b1', W1_out_dim, 1)

    sp.var('k1 k2 nu1 nu2 eta1 eta2 lambda a b x1 x2 eps')

    _in_ = sp.BlockMatrix([
        [sp.eye(W0_in_dim),      sp.zeros(W0_in_dim, W1_in_dim), sp.zeros(W0_in_dim, 1)],
        [sp.zeros(1, W0_in_dim), sp.zeros(1, W1_in_dim),         sp.eye(1)]
     ])

    M_in_P = sp.MatMul(_in_.T, P, _in_)

    assert(sp.Matrix(M_in_P).is_symmetric())

    if logging:
        sp.pprint(M_in_P)
        sp.pprint(M_in_P.as_explicit())
        sp.pprint(M_in_P.subs(P_vals).as_explicit())

    return sp.Matrix(M_in_P.subs(P_vals))

def _build_T(dim):
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


def _build_Q_for_relu(dim):
    # build quadratic relaxation matrix for the graph of ReLU
    # applied componentwise on a vector in R^dim

    T, Q_vars = _build_T(dim)

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

    return Q, Q_vars

def build_M_mid(Q, Q_vals, W0, b0, W1, b1, logging=False, activ_func='relu'):
    W0_in_dim =  W0.shape[1]
    W0_out_dim = W0.shape[0]
    W1_in_dim =  W1.shape[1]
    W1_out_dim = W1.shape[0]
    assert(W0_out_dim == W1_in_dim)
    assert(W0_out_dim == b0.shape[0])
    assert(W1_out_dim == b1.shape[0])
    assert(Q.shape[0] == W0_in_dim + W1_out_dim + 1)
    assert(Q.shape[0] == Q.shape[1])

    _W0 = sp.MatrixSymbol('W0', W0_out_dim, W0_in_dim)
    _b0 = sp.MatrixSymbol('b0', W0_out_dim, 1)
    _W1 = sp.MatrixSymbol('W1', W1_out_dim, W1_in_dim)
    _b1 = sp.MatrixSymbol('b1', W1_out_dim, 1)

    # sp.var('k1 k2 nu1 nu2 eta1 eta2 lambda a b x1 x2 eps')

    _mid_ = sp.BlockMatrix([
        [_W0,                              sp.zeros(W0_out_dim, W1_in_dim), _b0],
        [sp.zeros(W1_out_dim, W0_out_dim), sp.eye(W1_in_dim),               sp.zeros(W1_out_dim, 1)],
        [sp.zeros(1, W0_in_dim),           sp.zeros(1, W1_in_dim),          sp.eye(1, 1)],
     ])

    M_mid_Q = sp.MatMul(_mid_.T, Q, _mid_)

    assert(sp.Matrix(M_mid_Q).is_symmetric())

    vals={_W0: W0, _b0: b0}
    if logging:
        sp.pprint(M_mid_Q)
        sp.pprint(sp.Matrix(M_mid_Q.subs(Q_vals)))

    return sp.Matrix(M_mid_Q.subs(Q_vals))

def one_NN_SDP_builder():
    # build a 2 layer NN with 1x1 weight matrices
    # feed weights into matrix builder functions

    W0 = sp.Matrix([[.9]])
    b0 = sp.Matrix([[1]])
    W1 = sp.Matrix([[.8]])
    b1 = sp.Matrix([[3]])
    S  = sp.Matrix([[0, 0, -1],
                    [0, 0, 1],
                    [-1, 1, 0]])
    Q = S

    # sp.var('k1 k2 nu1 nu2 eta1 eta2 lambda a b x1 x2 eps')
    # p_vals = {x1:1, x2:1, eps:0.01, a:2, b:3}
    # P = sp.Matrix([[-2*a,   0,      2*a*x1],
                   # [0,      -2*b,   2*a*x2],
                   # [2*a*x1, 2*a*x2, (-2*(a*x1**2 + b*x2**2 + (a-b)*eps**2))]])

    # M_in_P = build_M_in(P=P, P_vals=p_vals, W0=W0, b0=b0, W1=W1, b1=b1, logging=True)
    M_out_S = build_M_out(S=S, W0=W0, b0=b0, W1=W1, b1=b1, logging=True)
    # M_mid_Q = build_M_mid(Q=Q, Q_vals=p_vals, W0=W0, b0=b0, W1=W1, b1=b1, logging=True)

def _build_quadratic_aprox_for_hypercube(center):
    # TODO: turn this into a function that generates a hypercube relaxation at a given point
    sp.var('a b x1 x2 eps')
    p_vals = {x1:1, x2:1, eps:0.01, a:2, b:3}
    # this is the form of a quadratic relaxation for a unit ball under infinity-norm
    # (aka hyper-cube) in R^2, with radius eps centered at (x1,x2). a,b are restricted to gt 0.
    P = sp.Matrix([[-2*a,   0,      2*a*x1],
                   [0,      -2*b,   2*a*x2],
                   [2*a*x1, 2*a*x2, (-2*(a*x1**2 + b*x2**2 + (a-b)*eps**2))]])
    return P, p_vals

def _relaxation_for_half_space(c, d):
    # TODO: make it a real function
    # for half space defined by cx - d <= 0
    # this quadratic relaxation defines the halfspace in R^2 defined by the line x1>x2
    S  = sp.Matrix([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 1],
                    [0, 0, -1, 1, 0]])
    return S

def two_NN_SDP_builder():
    # build a 2 layer NN with 2x2 weight matrices
    # feed weights into matrix builder functions

    # model = TwoLayerFFNet(2, 2, 2)
    # model.init_weights(w0=[[1,0],[0,1]], b0=[1,1],
                       # w1=[[2,0],[0,2]], b1=[2,2])
    # model.to(device)
    # print(model)

    W0 = sp.Matrix([[.9, 0], [0,.9]])
    b0 = sp.Matrix([[1],[1]])
    W1 = sp.Matrix([[.9, 0], [0,.9]])
    b1 = sp.Matrix([[1],[1]])


    S = _relaxation_for_half_space(None, None)
    P, p_vals = _build_quadratic_aprox_for_hypercube(center=None)
    M_in_P = build_M_in(P=P, P_vals=p_vals, W0=W0, b0=b0, W1=W1, b1=b1, logging=False)
    Q, Q_vars = _build_Q_for_relu(dim=W0.shape[0])
    M_mid_Q = build_M_mid(Q=Q, Q_vals=p_vals, W0=W0, b0=b0, W1=W1, b1=b1, logging=False)
    M_out_S = build_M_out(S=S, W0=W0, b0=b0, W1=W1, b1=b1, logging=False)

    X = M_in_P + M_mid_Q + M_out_S
    sp.pprint(X)

    # How to programatically convert to cvxpy problem?
    # sp.pprint(Q)
    # for i in range(0, Q.shape[0]):
        # for j in range(0, Q.shape[1]):
            # el = Q[i,j]
            # print(f"{el} - {type(el)}")


if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    two_NN_SDP_builder()
    # one_NN_SDP_builder()

    exit()
