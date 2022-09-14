from src.models.multi_layer import MultiLayerNN
from src.algorithms.certify import Certify, symbolic_relaxation_for_hypercube
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pylab as pyl
import math


def plot_quadratic_form(matrix):
    # plot a symetric matrix as an elipse
    # for a symetric matrix
    # [[a, b],
      # b, c]]
    matrix = np.array(matrix)
    # assert matrix.shape == (2,2)
    assert matrix[0,1] == matrix[1, 0]
    matrix = np.array(matrix)
    a = matrix[0,0]
    b = matrix[0,1]
    c = matrix[1,1]
    lambda1 = (a + c)/2 + math.sqrt( (a-c)**2/4 + b**2 )
    lambda2 = (a + c)/2 - math.sqrt( (a-c)**2/4 + b**2 )
    if b==0 and a >= c:
        theta = 0
    elif b==0 and a < c:
        theta = math.pi / 2
    else:
        theta = math.atan2( (lambda1 - a), b )
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    # e = Ellipse((0,0), lambda1, lambda2, theta, alpha=0.4)
    print(f"{lambda1}, {lambda2}, {theta}")
    e = Ellipse((0,0), lambda1, lambda2, theta)
    ax.add_artist(e)
    x = max(lambda1, lambda2)
    ax.set_xlim(-x, x)
    ax.set_ylim(-x, x)
    plt.show()


    print(matrix)

def test_plot():
    x = [0,1,0]
    y = [0,0,1]

    plt.figure(figsize=(4,4))
    plt.axis('equal')
    plt.fill(x,y)
    plt.show()


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
    # tl = x + np.array([[-eps], [eps]])
    # tr = x + np.array([[eps], [eps]])
    # br = x + np.array([[eps], [-eps]])
    # bl = x + np.array([[-eps], [-eps]])
    # x1, x2 = zip(*[tl, tr, bl, br])
    # print(x1)
    # print(x2)
    # plt.figure(figsize=(8, 8))
    # plt.axis('equal')
    # plt.fill(x1, x2)

    p = make_quadratic(P)
    grid_size = 500
    _x, _y = np.linspace(-10,10,grid_size), np.linspace(-10,10,grid_size)
    x, y  = np.meshgrid(_x, _y)
    z = p(x,y)

    im = pyl.imshow(z,cmap=pyl.cm.RdBu) # drawing the function
    pyl.show()


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
        print(P)
        print_P(P, x, eps)
        # plot_quadratic_form(P)

    # m = [[1, 3, 0],
         # [3, 2, 0],
         # [0, 0, 1]]
    # exit()
