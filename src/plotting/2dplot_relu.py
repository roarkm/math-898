from src.models.multi_layer import (MultiLayerNN,
                                    identity_map)
from src.algorithms.certify import (Certify,
                                    _relaxation_for_hypercube,
                                    _relaxation_for_half_space)
from matplotlib.patches import Ellipse
from src.algorithms.abstract_verifier import (constraints_for_separating_hyperplane,
                                              # str_constraints,
                                              constraints_for_inf_ball)

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cvxpy as cp
import numpy as np
import math
import sys
import logging


def plot_relu_for_constrained_region(constraints, free_vars,
                                     values={'set':1, 'img':1, 'not_set':0, 'not_img':0},
                                     colors={'bkgrnd':  (0.0, 0.0, 0.0, 0.0),
                                             'set':     (0.0, 0.0, 1.0, 1.0),
                                             'img':     (0.0, 0.0, 0.0, 1.0),
                                             'not_set': (0.0, 0.0, 0.0, 0.0),
                                             'not_img': (0.0, 0.0, 0.0, 0.0)},
                                     resolution=0.05,
                                     plot_boundaries={'t':10, 'r':10, 'b':-10, 'l':-10},
                                     plot_scale=1):
    # plot the region satisfying constraints
    # plot the image under relu of the region satisfying constraints
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
    # plt.axhline(linestyle='--')
    # plt.axvline(linestyle='--')
    plt.xlabel('x1')
    plt.ylabel('x2')

    pre_img = make_set_indicator(constr, free_vars, values=values)
    z = pre_img(x1, x2)
    cmap = ListedColormap([colors['not_set'], colors['set']])
    plt.imshow(z, interpolation='none',
               extent=extent, origin='lower', cmap=cmap)

    max_x1, max_x2 = max_in_region(constr, free_vars, x1, x2)
    relu_img = make_relu_img_indicator(constr, free_vars, max_x1, max_x2, values=values)
    img_z = relu_img(x1, x2)
    cmap = ListedColormap([colors['not_img'], colors['img']])
    plt.imshow(img_z, interpolation='none',
               extent=extent, origin='lower', cmap=cmap)
    plt.show()


def make_set_indicator(constraints, free_vars, values={'set':1, 'not_set':0}):
    # indicator function for the region satisfying constraints
    expr = cp.transforms.indicator(constraints)
    def indicator(x1, x2):
        free_vars.value = np.array([[x1], [x2]])
        if expr.value == 0:
            return values['set']
        return values['not_set']
    return np.vectorize(indicator)


def min_in_region(pre_img_constrs, free_vars, _x1, _x2):
    # create a function that will return the minimum
    # x1, x2 that satisfy constraints
    expr = cp.transforms.indicator(pre_img_constrs)

    def min_of(x1, x2, mins):
        # mutates mins
        free_vars.value = np.array([[x1], [x2]])
        if expr.value == 0:
            if x2 <= mins['x2']:
                mins['x2'] = x2
            if x1 <= mins['x1']:
                mins['x1'] = x1
    m = np.vectorize(max_of)

    mins = {'x1': sys.float_info.max, 'x2': sys.float_info.max}
    m(_x1, _x2, mins)
    return mins['x1'], mins['x2']


def max_in_region(pre_img_constrs, free_vars, _x1, _x2):
    # create a function that will return the maximum
    # x1, x2 that satisfy constraints and are in the non-pos orthant
    expr = cp.transforms.indicator(pre_img_constrs)
    def max_of(x1, x2, maxes):
        # mutates maxes
        if x1 <= 0 and x2 >= maxes['x2']:
            free_vars.value = np.array([[x1], [x2]])
            if expr.value == 0:
                maxes['x2'] = x2
        if x2 <= 0 and x1 >= maxes['x1']:
            free_vars.value = np.array([[x1], [x2]])
            if expr.value == 0:
                maxes['x1'] = x1
    m = np.vectorize(max_of)

    maxes = {'x1':0, 'x2':0}
    m(_x1, _x2, maxes)
    return maxes['x1'], maxes['x2']


def make_relu_img_indicator(pre_img_constrs, free_vars, max_x1, max_x2,
                            values={'img':1, 'not_img':0}):
    # indicator function for the image under relu
    # of the region satisfying pre_img_constrs
    expr = cp.transforms.indicator(pre_img_constrs)
    def indicator(x1, x2):
        tol = 0.01
        if x1 > 0 and x2 > 0:
            free_vars.value = np.array([[x1], [x2]])
            if expr.value == 0:
                return values['img']
            if x1 <= tol and x2 <= max_x2 and x2 >= 0:
                return values['img']
            if x2 <= tol and x1 <= max_x1 and x1 >= 0:
                return values['img']
            return values['not_img']
        if x1 <= 0 and x1 >= -tol and x2 <= max_x2 and x2 >= 0:
            return values['img']
        if x2 <= 0 and x2 >= -tol and x1 <= max_x1 and x1 >= 0:
            return values['img']
        return values['not_img']
    return np.vectorize(indicator)


def plot_relu_R1(resolution=0.01):
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


def constraints_for_polytope():
    # custom polytope
    A = np.array([
        [2,  1],
        [-1, 0],
        [0, -1]
    ])
    b = np.array([
        [1, 2, 2]
    ])
    _x = cp.Variable((2, 1))
    return [A @ _x <= b], _x


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    center = np.array([[2], [1]])
    eps = 1
    constr, free_vars = constraints_for_polytope()
    # logging.info(str_constraints(constr))
    plot_relu_for_constrained_region(constr, free_vars,
                                     plot_boundaries={'t':3.5, 'r':1, 'b':-2, 'l':-1},
                                     plot_scale=1, resolution=0.02)
    exit()

    # g_vals = np.array([9, 2])
    g_vals = 100 * np.random.random_sample(center.shape[0])
    P, _, _ = _relaxation_for_hypercube(center, eps, values=g_vals)
    exit()

    c = np.array([[-1], [1]])
    d = 0
    S = _relaxation_for_half_space(c,d,2)
    S = S[np.ix_([2,3,4], [2,3,4])] # get submatrix corresponding to output of nn
    plot_half_space(c, d, relaxation_matrix=S) # relaxation is tight so not visible
