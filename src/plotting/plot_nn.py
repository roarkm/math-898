from src.models.multi_layer import MultiLayerNN, identity_map
import matplotlib.pyplot as plt
import cvxpy as cp
import torch
import numpy as np


def plot_nn(f, resolution=0.1):
    _x1 = np.arange(-1, 10, resolution)
    _x2 = np.arange(-1, 10, resolution)
    x1, x2  = np.meshgrid(_x1, _x2)

    f_map = make_nn_indicator(f)
    z = f_map(x1, x2)

    plt.grid(True)
    plt.axhline()
    plt.axvline()

    plt.contourf(x1, x2, z)

    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def make_nn_indicator(f, values=[0, 1]):
    # 2d only
    def indicator(x1, x2):
        im_x = f(torch.tensor([[x1], [x2]]).T.float()).detach().numpy()
        _class = np.argmax(im_x)
        return values[_class]
    return np.vectorize(indicator)


if __name__ == '__main__':
    f = identity_map(2,2)
    plot_nn(f)
