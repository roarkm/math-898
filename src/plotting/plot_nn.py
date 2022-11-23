from src.models.multi_layer import (MultiLayerNN,
                                    identity_map,
                                    custom_net)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cvxpy as cp
import torch
import numpy as np
import math


def plot_nn(f, resolution=0.1, x_range=[-1,14], y_range=[-1,10]):
    _x1 = np.arange(x_range[0], x_range[1], resolution)
    _x2 = np.arange(y_range[0], y_range[1], resolution)
    x1, x2  = np.meshgrid(_x1, _x2)

    f_map = make_nn_indicator(f)
    z = f_map(x1, x2)

    extent = np.min(x1), np.max(x1), np.min(x2), np.max(x2)
    fig, ax = plt.subplots()
    ax.axis('equal')
    # plt.axhline(linestyle='--')
    # plt.axvline(linestyle='--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)

    plt.axhline()
    plt.axvline()

    cmap = ListedColormap([(0.0,1.0,0.0,0.5), (0.0,0.0,1.0,0.5)])
    plt.imshow(z, interpolation='none',
               extent=extent, origin='lower', cmap=cmap)
    plt.show()

def make_nn_indicator(f, values=[0, 1]):
    # 2d only
    def indicator(x1, x2):
        im_x = f(torch.tensor([[x1], [x2]]).T.float()).detach().numpy()
        _class = np.argmax(im_x)
        return values[_class]
    return np.vectorize(indicator)

def rotation_mat(angle):
    # rotates by angle radians in clockwise direction
    R = [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle), math.cos(angle)]]
    return R

def custom_net():
    weights = [ ]
    weights.append(rotation_mat(0.35))
    weights.append(rotation_mat(-0.25))
    # weights.append(rotation_mat(-0.9))
    # weights.append(rotation_mat(0.2))
    # # weights.append([[0.3,0],
                    # # [0,0.3]])
    bias_vecs =[
        [-3,-1],
        [-5,-1],
        # [-3,-1],
        # [-1,-1],
        # [0, -2]
    ]
    print(weights)
    # exit()
    return MultiLayerNN(weights, bias_vecs)

if __name__ == '__main__':
    f = custom_net()
    plot_nn(f)
