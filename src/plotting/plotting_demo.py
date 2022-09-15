from src.models.multi_layer import MultiLayerNN
from src.algorithms.certify import Certify, symbolic_relaxation_for_hypercube
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def make_quadratic(P):
    # returns a function that computes the quadratic invoked by P
    def M_quad(x1, x2):
        _P = np.array(P)
        # assert len(x)+1 == _M.shape[1]
        _x = np.vstack(np.concatenate([[x1], [x2], np.array([1])]))
        z =  _x.T @ _P @ _x
        if z >= 0:
            return 1
        return 0
    return np.vectorize(M_quad)


def print_P(P, x=None, eps=None):
    # TODO: plot inf-ball around x
    # tl = x + np.array([[-eps], [eps]])
    # tr = x + np.array([[eps], [eps]])
    # br = x + np.array([[eps], [-eps]])
    # bl = x + np.array([[-eps], [-eps]])
    p = make_quadratic(P)
    # TODO: detect min, max values of z where z == 1 (use to set plot xlim, ylim)
    # TODO: set colors, labels
    # TODO: figure out explicit ellipse formula (and plot elipse instead?)
    # TODO: plot safety set relaxation too (subplot)
    # QUESTION: Q matrix encapsulates all middel layers into one big,
    #           but does each block along the diag correspond to a layer
    #           relaxation!?
    grid_size = 100
    _x1, _x2 = np.arange(6, 12, 0.01), np.arange(-1, 1, 0.01)
    x1, x2  = np.meshgrid(_x1, _x2)
    z = p(x1, x2)
    plt.contourf(x1, x2, z)
    plt.plot(x[0][0], x[1][0], marker="o", markersize=20) # plot x
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()



if __name__ == '__main__':
    weights = [
        [[1, 0],
         [0, 1]],
        [[1, 0],
         [0, 1]],
    ]
    bias_vecs =[
        (0,0),
        (0,0),
    ]
    f = MultiLayerNN(weights, bias_vecs)
    cert = Certify(f)
    eps = 0.5
    x = [[9], [0]]
    is_robust = cert.verifiy_at_point(x=x, eps=eps)
    if is_robust:
        P = cert.P
        # print(P)
        gamma1 = P[0,0] / (-2)
        gamma2 = P[1,1] / (-2)
        assert P[2,0] == 18*gamma1
        assert P[2,2] == -2*(8.5*9.5*gamma1 + -0.5*0.5*gamma2)
        # print(f"gamma 1: {gamma1}, gamma 2: {gamma2}")
        _qp = make_quadratic(P)
        assert _qp(9,0) >= 0
        assert _qp(8.5,0.5) >= 0
        assert _qp(9.5,0.5) >= 0
        assert _qp(9.5,-0.5) >= 0
        assert _qp(8.5,-0.5) >= 0

        print_P(P, x, eps)

# The below code snippet demonstrates voxel based 3d printing
# https://matplotlib.org/3.4.3/gallery/mplot3d/voxels_rgb.html#sphx-glr-gallery-mplot3d-voxels-rgb-py

# def midpoints(x):
    # sl = ()
    # for i in range(x.ndim):
        # x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        # sl += np.index_exp[:]
    # return x

# # prepare some coordinates, and attach rgb values to each
# r, g, b = np.indices((17, 17, 17)) / 16.0
# rc = midpoints(r)
# gc = midpoints(g)
# bc = midpoints(b)

# # define a sphere about [0.5, 0.5, 0.5]
# sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

# # combine the color components
# colors = np.zeros(sphere.shape + (3,))
# colors[..., 0] = rc
# colors[..., 1] = gc
# colors[..., 2] = bc

# # and plot everything
# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(r, g, b, sphere,
          # facecolors=colors,
          # edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
          # linewidth=0.5)
# ax.set(xlabel='r', ylabel='g', zlabel='b')

# plt.show()
