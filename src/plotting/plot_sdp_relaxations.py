from src.models.multi_layer import MultiLayerNN, identity_map
from src.algorithms.certify import Certify, _relaxation_for_hypercube, _relaxation_for_half_space
from matplotlib.patches import Ellipse
from src.algorithms.abstract_verifier import constraints_for_separating_hyperplane, str_constraints, constraints_for_inf_ball

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import math

def plot_half_space(c=[[-1], [1]], d=0, resolution=0.05,
                    relaxation_matrix=None, values={'set':1, 'not_set':0}):
    # 2d only
    c = np.array(c)
    half_space_ind = make_half_space_indicator(c,d)

    # TODO: parameterize the canvas size
    _x1 = np.arange(-1, 10, resolution)
    _x2 = np.arange(-1, 10, resolution)
    x1, x2  = np.meshgrid(_x1, _x2)

    test_point = np.array([[9], [0]])
    # assert half_space_ind(9, 0) == 1
    # assert half_space_ind(0, 9) == 0
    # exit()

    z = half_space_ind(x1, x2)
    # plt.grid(True)
    plt.axhline()
    plt.axvline()
    plt.contourf(x1, x2, z)

    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')

    if relaxation_matrix is not None:
        _qp = make_quadratic(relaxation_matrix)
        z = _qp(x1, x2)
        plt.contourf(x1, x2, z, alpha=0.2)
    plt.show()


def plot_inf_ball(center, eps, values={'set':1, 'not_set':0},
                  resolution=0.05, relaxation_matrix=None):
    assert center.shape[0] == 2

    # free_vars = cp.Variable((len(center),1), name='z')
    # constr, free_vars = constraints_for_inf_ball(center, eps, free_vars=free_vars)
    constr, free_vars = constraints_for_inf_ball(center, eps)

    ball_ind = make_set_indicator(constr, free_vars, values=values)

    # TODO: parameterize the canvas size
    _x1, _x2 = np.arange(-1, 3, resolution), np.arange(-1, 3, resolution)
    x1, x2  = np.meshgrid(_x1, _x2)

    z = ball_ind(x1, x2)
    # plt.grid(True)
    plt.axhline()
    plt.axvline()
    plt.contourf(x1, x2, z)
    plt.plot(center[0][0], center[1][0], marker="x", markersize=10)
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')

    if relaxation_matrix is not None:
        _qp = make_quadratic(relaxation_matrix)
        z = _qp(x1, x2)
        plt.contourf(x1, x2, z, alpha=0.2)
    plt.show()


def make_set_indicator(constraints, free_vars, values={'set':1, 'not_set':0}):
    # 2d only
    expr = cp.transforms.indicator(constraints)
    def indicator(x1, x2):
        free_vars.value = np.array([[x1], [x2]])
        if expr.value == 0:
            return values['set'] # (x1, x2) satisfy the constraints
        return values['not_set'] # (x1, x2) DO NOT satisfy the constraints

    return np.vectorize(indicator)


def make_half_space_indicator(c, d=0, values={'set':1, 'not_set':0}):
    # 2d only
    def indicator(x1, x2):
        if (np.dot(c.T, np.array([[x1], [x2]])) > d):
            return values['set'] # (x1, x2) satisfy the constraints
        return values['not_set'] # (x1, x2) DO NOT satisfy the constraints
    return np.vectorize(indicator)


def make_quadratic(P):
    # returns a function that computes the quadratic invoked by P
    def M_quad(x1, x2):
        _P = np.array(P)
        _x = np.vstack(np.concatenate([[x1], [x2], np.array([1])]))
        z =  _x.T @ _P @ _x
        if z >= 0:
            return 1
        return 0
    return np.vectorize(M_quad)


if __name__ == '__main__':
    # center = np.array([[1], [1]])
    # eps = 0.8
    # # g_vals = np.array([9, 2])
    # g_vals = 100 * np.random.random_sample(center.shape[0])
    # P, _, _ = _relaxation_for_hypercube(center, eps, values=g_vals)
    # plot_inf_ball(center, eps, resolution=0.05, relaxation_matrix=P.value)
    # exit()

    c = np.array([[-1], [1]])
    d = 0
    S = _relaxation_for_half_space(c,d,2)
    S = S[np.ix_([2,3,4], [2,3,4])] # get submatrix corresponding to output of nn
    plot_half_space(c, d, relaxation_matrix=S) # relaxation is tight so not visible
