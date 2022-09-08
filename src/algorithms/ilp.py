import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
# from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
from src.models.multi_layer import MultiLayerNN
from src.algorithms.abstract_verifier import AbstractVerifier


class IteratedLinearVerifier(AbstractVerifier):


    def __init__(self, f=None):
        super(IteratedLinearVerifier, self).__init__(f)
        self.constraints = list()


    def constraints_for_inf_ball(self, center, eps, verbose=False):
        # create a list containing the constraints for an inf-ball of radius eps
        # set constraints for hypercube around x
        A_lt = np.zeros((2, len(center)))
        a_vec = np.zeros((2, 1))

        z0 = cp.bmat([cp.Variable(len(center), name='z0')]).T
        A_lt[0][0] = 1
        a_vec[0][0] = center[0][0] + eps
        A_lt[1][0] = -1
        a_vec[1][0] = -1 * (center[0][0] - eps)

        if len(center) > 1:
            for i in range(1, len(center)):
                row = np.zeros((1, len(center)))
                row[0][i] = 1
                A_lt = np.vstack([A_lt, row])
                _a = np.matrix([[ float(center[i][0]) + eps ]])
                a_vec = np.vstack([a_vec, _a])
                row = np.zeros((1, len(center)))
                row[0][i] = -1
                A_lt = np.vstack([A_lt, row])
                _a = np.matrix([[ -1 * float(center[i][0]) - eps ]])
                a_vec = np.vstack([a_vec, _a])

        if verbose:
            print(f"Constraints for inf-ball radius {eps} at center {center}")
            print(A_lt)
            print(a_vec)
        return [A_lt @ z0 <= a_vec]


    def constraints_for_k_class_polytope(self, k, x, verbose=False):
        # build a polytope constraint matrix corresponding to the output region
        # of the k-th class
        # ie - the polytope corresponding to the region of R^dim where
        # the k-th component is greater than all other components
        # returns a list containing the inequality constraints in matrix form
        z = cp.bmat([cp.Variable(len(x), name='z')]).T
        n_rows = len(x)-1
        A = np.zeros((n_rows, len(x)))
        A[:,k] = 1

        row = 0
        for j in range(0, len(x)):
            if j == k:
                continue
            A[row][j] = -1
            row += 1

        b = np.zeros((n_rows, 1))
        if verbose:
            print(f"Polytope for {len(x)}-class classifier. k={k}.")
            print(A)
            print(b)
        return [A @ z >= b] # for compliment, use <= ?

    def verify_at_point(self, x=[[9], [0]], eps=0.5):
        # use ILP to verify all x' within eps inf norm of x
        # are classified the same as x'
        # (ie: f(x) == f(x') for all x' in inf-norm ball centered at x
        assert self.f != None, "No NN provided."
        assert self.nn_weights[0].shape[1] == len(x), "x is the wrong shape"

        # creat objective function: inf-norm(x - z) with z as opt var, x fixed.
        # n = cp.norm(x, 'inf')

        x = np.matrix(x)
        im_x = self.f(torch.tensor(x).T.float()).data.T.tolist()
        x_class = np.argmax(im_x)

        constraints = []
        # set constraints for hypercube around x
        constraints += self.constraints_for_inf_ball(center=x, eps=eps)
        constraints += self.constraints_for_k_class_polytope(k=0, x=x)

        exit()

        # propogate x layer by layer through f adding constraints based on activation pattern
        _im_x = np.matrix(x)
        for i, w in enumerate(self.nn_weights):
            _wi = np.matrix(w)
            _bi = np.matrix(self.nn_bias_vecs[i])
            _im_x = _wi * _im_x + _bi.T

            # zi = cp.Variable(_wi.shape[0], f"z{i}")

            # if i > 0:
                # self.constraints += [zi == _wi * zi + _bi]

            # if i < len(self.nn_weights):
                # _im_x = self.relu(torch.tensor(_im_x)).numpy()

        print(_im_x)

        # do this on last layer
        # set constraints for Y or its compliment (halfspace safety set)
        assert len(im_x) == 2, "Only binary classifiers currently supported"
        d = 0
        c = np.matrix('-1; 1') # defines halfspace where y1 > y2
        if x_class == 1:
            c = -1 * c # defines halfspace where y2 > y1

        exit()

        # add constraints eg
        # self.constraints += [_nus >= 0]

        # prob = cp.Problem(cp.Minimize(1), constraints)
        # prob.solve()

        return None


if __name__ == '__main__':

    weights = [
        [[1, 0],
         [0, 1]],
        [[2, 0],
         [0, 2]],
        # [[3, 0],
         # [0, 3]]
    ]
    bias_vecs =[
        (1,1),
        (2,2),
        # (3,3),
    ]
    f = MultiLayerNN(weights, bias_vecs)
    ilp = IteratedLinearVerifier(f)
    ilp.verify_at_point()
    # print(ilp)
    # iterated_linear_program(x=[[9],[1]], eps=0.007, f=f)
