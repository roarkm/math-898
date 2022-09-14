from src.models.multi_layer import MultiLayerNN
from src.algorithms.certify import Certify
import matplotlib.pyplot as plt
import numpy as np


def do_certify():
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
    x = [[9], [0]]
    eps = 0.5
    is_robust = cert.verifiy_at_point(x=x, eps=eps, verbose=True)
    print(f"Identity is {eps}-robust at {x}? {is_robust}")


def plot_quadratic_form(matrix):
    # plot a symetric matrix as an elipse
    # if two
    matrix = np.array(matrix)
    print(matrix)

def test_plot():
    x = [0,1,0]
    y = [0,0,1]

    plt.figure(figsize=(4,4))
    plt.axis('equal')
    plt.fill(x,y)
    plt.show()


def make_quadratic(M):
    # returns a function that computes the quadratic invoked by P
    def M_quad(x,y):
        _M = np.array(M.value)
        # assert len(x)+1 == _M.shape[1]
        _x = np.vstack(np.concatenate([[x], [y], np.array([1])]))
        z =  _x.T @ _M @ _x
        # print(z[0][0])
        # print((z+1)[0][0]
        # z =  z + 1
        # z =  np.maximum(0, _x.T @ _M @ _x + 8)
        return z
    return np.vectorize(M_quad)
    # return M_quad


def print_P(P, ref_point, eps):

    p = make_quadratic(P)
    grid_size = 50
    _x, _y = np.linspace(-10,10,grid_size), np.linspace(-10,10,grid_size)
    x, y  = np.meshgrid(_x, _y)
    z = p(x,y)

    im = pyl.imshow(z,cmap=pyl.cm.RdBu) # drawing the function
    cset = pyl.contour(z, np.arange(z.min(), z.max(), (z.max() - z.min())/10),
                       linewidths=2, cmap=pyl.cm.Set2) # adding the Contour lines with labels
    pyl.clabel(cset, inline=True, fmt='%1.1f',fontsize=10)
    pyl.colorbar(im) # adding the colobar on the right
    # latex fashion title
    pyl.show()

    # h = plt.contourf(x, y, z)
    # plt.axis('scaled')
    # plt.colorbar()
    # plt.show()

    # fig = plt.figure(figsize = (10,7))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap='cool')
    # ax.set_title("Surface Bonus", fontsize = 13)
    # ax.set_xlabel('x', fontsize = 11)
    # ax.set_ylabel('y', fontsize = 11)
    # ax.set_zlabel('Z', fontsize = 11)
    # plt.show()

if __name__ == '__main__':
    m = [[1, 2],
         [2, 3]]
    plot_quadratic_form(m)
