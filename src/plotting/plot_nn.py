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
import math


def plot_nn(f, resolution=0.1, x_range=[-1,10], y_range=[-1,6], x=None, eps=None):
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

    add_class_regions(f, x1, x2, extent)

    # np.random.seed(2)
    # np.random.seed(179)
    # np.random.seed(179)
    # g_vals = 100 * np.random.random_sample(_x.shape[0])
    _x = np.array([[x[0][0]], [x[1][0]]])
    g_vals = np.array([9, 2])
    # g_vals = np.array([2, 9])
    P, _, _ = _relaxation_for_hypercube(_x, eps, values=g_vals)

    add_relaxation(P.value, x1, x2, extent)

    if x is not None:
        add_inf_ball(x[0][0], x[1][0], eps, ax)
    plt.show()


if __name__ == '__main__':
    f = custom_net()
    plot_nn(f, x=[[4], [2]], eps=1, resolution=0.1)
