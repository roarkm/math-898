from src.models.multi_layer import (MultiLayerNN,
                                    identity_map,
                                    custom_net)
from src.algorithms.abstract_verifier import constraints_for_inf_ball
from src.algorithms.certify import _relaxation_for_hypercube
from src.plotting.common import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import cvxpy as cp
import numpy as np
import torch
import math


c1 = (0.0,1.0,0.0,0.5)
c2 = (0.0,0.0,1.0,0.5)

def make_quadratic(P):
    # returns a function that computes the quadratic invoked by P
    _P = np.array(P)
    def M_quad(x1, x2):
        _x = np.vstack(np.concatenate([[x1], [x2], np.array([1])]))
        z =  _x.T @ _P @ _x
        if z >= 0:
            return 1
        return 0
    return np.vectorize(M_quad)


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


def custom_polytope_constraints():
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


def add_inf_ball(x1, x2, eps, ax):
    # plot x
    plt.plot(x1, x2, marker='o', markersize=7, color='blue')
    ax.annotate(f"$x^0$ = ({x1}, {x2})", (x1+0.3, x2+0.2))

    # plot inf ball border
    tl = [x1 - eps, x2 + eps]
    tr = [x1 + eps, x2 + eps]
    bl = [x1 - eps, x2 - eps]
    br = [x1 + eps, x2 - eps]
    plt.plot([tl[0], tr[0]], [tl[1], tl[1]], linestyle="--", color='blue')
    plt.plot([bl[0], br[0]], [bl[1], bl[1]], linestyle="--", color='blue')
    plt.plot([bl[0], tl[0]], [bl[1], tl[1]], linestyle="--", color='blue')
    plt.plot([br[0], tr[0]], [br[1], tr[1]], linestyle="--", color='blue')
    ball_str = "$\mathcal{B}_{\infty}("+str(eps)+",x^0)$"
    plt.text(tl[0], tl[1]+0.3, ball_str)


def add_class_regions(f, x1r, x2r, extent,
                      colors=[(0.0,1.0,0.0,0.5), (0.0,0.0,1.0,0.5)]):

    f_map = make_nn_indicator(f)
    z = f_map(x1r, x2r)
    cmap = ListedColormap(colors)
    im = plt.imshow(z, interpolation='none',
                    extent=extent, origin='lower',
                    cmap=cmap)

    values = np.unique(z)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i],
                              label="Class {l}".format(l=values[i]+1)) for i in range(len(values))]
    plt.legend(handles=patches,
               loc=1, borderaxespad=1.0,)


def add_relaxation(P, x1, x2, extent,
                   colors=[(0.0,0.0,0.0,0.0), (0.0,1.0,1.0,0.5)]):
    _qp = make_quadratic(P)
    z = _qp(x1, x2)
    cmap = ListedColormap(colors)
    im = plt.imshow(z, interpolation='none',
                    extent=extent, origin='lower', cmap=cmap)


def make_nn_indicator(f, values=[0, 1]):
    # 2d only
    def indicator(x1, x2):
        im_x = f(torch.tensor([[x1], [x2]]).T.float()).detach().numpy()
        _class = np.argmax(im_x)
        return values[_class]
    return np.vectorize(indicator)


