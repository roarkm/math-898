from src.models.multi_layer import MultiLayerNN, identity_map
from src.algorithms.certify import (Certify,
                                    _relaxation_for_hypercube,
                                    _relaxation_for_half_space)
from matplotlib.patches import Ellipse
from src.plotting.common import *
from src.plotting.plot_nn import (add_inf_ball,
                                  add_relaxation)
from src.algorithms.abstract_verifier import (constraints_for_separating_hyperplane,
                                              constraints_for_inf_ball)

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
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

    z = half_space_ind(x1, x2)
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


def plot_inf_ball(c, eps, values={'set':1, 'not_set':0},
                  x_range=[-2,4], y_range=[-2,4],
                  resolution=0.1, relaxation_matrix=None):

    _x1 = np.arange(x_range[0], x_range[1], resolution)
    _x2 = np.arange(y_range[0], y_range[1], resolution)
    x1, x2  = np.meshgrid(_x1, _x2)
    pad = 2
    extent = np.min(x1)-pad, np.max(x1)+pad, np.min(x2)-pad, np.max(x2)+pad

    fig, ax = plt.subplots()
    ax.axis('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)
    plt.axhline()
    plt.axvline()

    add_inf_ball(c[0][0], c[1][0], eps, ax)

    if relaxation_matrix is not None:
        add_relaxation(relaxation_matrix, x1, x2, extent)

    plt.show()
    return


def plot_relu(resolution=0.01):
    # TODO: parameterize the canvas size
    x = np.arange(-5, 8, resolution)

    y = np.maximum(0, x) #

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # plt.grid(True)
    # ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x, y, 'r')

    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.axhline()
    # plt.axvline()

    # show the plot
    plt.show()


def plot_constrained_region(constraints, free_vars,
                            values={'set':1, 'not_set':0},
                            colors={'bkgrnd':  (0.0, 0.0, 0.0, 0.0),
                                    'set':     (0.0, 0.0, 1.0, 1.0),
                                    'not_set': (0.0, 0.0, 0.0, 0.0)},
                            resolution=0.05,
                            plot_boundaries={'t':10, 'r':10, 'b':-10, 'l':-10},
                            plot_scale=1):
    _x1 = np.arange(plot_boundaries['l']*plot_scale,
                    plot_boundaries['r']*plot_scale,
                    resolution)
    _x2 = np.arange(plot_boundaries['b']*plot_scale,
                    plot_boundaries['t']*plot_scale,
                    resolution)
    x1, x2  = np.meshgrid(_x1, _x2)
    extent = np.min(x1), np.max(x1), np.min(x2), np.max(x2)

    fig, ax = plt.subplots()
    ax.axis('equal')
    plt.xlabel('x1')
    plt.ylabel('x2')

    pre_img = make_set_indicator(constraints, free_vars, values=values)
    z = pre_img(x1, x2)
    cmap = ListedColormap([colors['not_set'], colors['set']])
    plt.imshow(z, interpolation='none',
               extent=extent, origin='lower', cmap=cmap)
    plt.show()


if __name__ == '__main__':
    # pt, fvars = custom_polytope_constraints()
    # plot_constrained_region(pt, fvars)

    # plot_relu()
    # exit()

    center = np.array([[1], [1]])
    eps = 0.8
    g_vals = np.array([9, 2])
    # np.random.seed(2)
    # np.random.seed(179)
    # g_vals = 100 * np.random.random_sample(center.shape[0])

    P, _, _ = _relaxation_for_hypercube(center, eps, values=g_vals)
    plot_inf_ball(center, eps, resolution=0.05, relaxation_matrix=P.value)
    # exit()

    # c = np.array([[-1], [1]])
    # d = 0
    # S = _relaxation_for_half_space(c,d,2)
    # S = S[np.ix_([2,3,4], [2,3,4])] get submatrix corresponding to output of nn
    # plot_half_space(c, d, relaxation_matrix=S) relaxation is tight so not visible
