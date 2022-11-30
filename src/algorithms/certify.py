from src.algorithms.certify_relu import CertifyReLU
from src.algorithms.certify_affine import CertifyAffine
from src.models.multi_layer import (identity_map,
                                    MultiLayerNN)
import numpy as np


def Certify(f):
    if f.final_relu:
        return CertifyReLU(f)
    else:
        return CertifyAffine(f)


def quick_test_eps_robustness():
    f = identity_map(2, 2)
    x = [[9], [1]]
    cert = Certify(f)
    eps = 1
    e_robust = cert.decide_eps_robustness(x, eps, verbose=False)

    x_class = f.class_for_input(x)
    print(f"f({x}) = class {x_class+1}")
    print(f"{f.name} is ({eps})-robust at {x}?  {e_robust}")


def symbolic_test():
    b = [
        np.array([[1],
                  [1]]),
        np.array([[2],
                  [2],
                  [2]]),
        np.array([[3],
                  [3]]),
    ]
    weights = [
        np.array([[1, 1, 1],
                  [1, 1, 1]]),
        np.array([[2, 2],
                  [2, 2],
                  [2, 2]]),
        np.array([[3, 3, 3],
                  [3, 3, 3]]),
    ]
    f = MultiLayerNN(weights, b)
    # f = identity_map(2, 2)
    cert = Certify(f)
    eps = 8
    x = [[9], [1], [1]]
    cert.build_symbolic_matrices(x, eps)


if __name__ == '__main__':
    quick_test_eps_robustness()
    # symbolic_test()
