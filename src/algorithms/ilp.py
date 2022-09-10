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

        # top constraint
        z0 = cp.bmat([cp.Variable(len(center), name='z0')]).T
        A_lt[0][0] = 1
        a_vec[0][0] = center[0][0] + eps
        # bottom constraint
        A_lt[1][0] = -1
        a_vec[1][0] = -1 * (center[0][0] - eps)

        if len(center) > 1:
            for i in range(1, len(center)):
                # top constraint
                row = np.zeros((1, len(center)))
                row[0][i] = 1
                A_lt = np.vstack([A_lt, row])
                _a = np.matrix([[ float(center[i][0]) + eps ]])
                a_vec = np.vstack([a_vec, _a])

                # bottom constraint
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


    def verify_at_point(self, x=[[9], [0]], eps=0.5):
        # use ILP to verify all x' within eps inf norm of x
        # are classified the same as x'
        # (ie: f(x) == f(x') for all x' in inf-norm ball centered at x
        assert self.f != None, "No NN provided."
        assert self.nn_weights[0].shape[1] == len(x), "x is the wrong shape"

        _im_x = np.matrix(x)
        fx = self.f(torch.tensor(_im_x).T.float()).detach().numpy()
        x_class = np.argmax(fx)

        # optimization var list starting with x0
        # x0 is unconstrained since we are using inf-ball in objective function
        xi_list = [cp.bmat([cp.Variable(len(x), name='x0')]).T]

        # propogate x layer by layer through f
        # 1) add constraints for affine transforms
        # 2) add constraints for ReLU activation pattern
        for i in range(0, len(self.nn_weights)):
            Wi = np.matrix(self.nn_weights[i])
            bi = np.matrix(self.nn_bias_vecs[i]).T

            # add constraint for affine transformation
            xi_list.append(cp.bmat([cp.Variable(Wi.shape[0], f"x{i}_hat")]).T)
            xi_hat = Wi @ xi_list[-2] + bi
            self.constraints += [xi_list[-1] == xi_hat]

            # add constraints for ReLU activation pattern
            xi_list.append(cp.bmat([cp.Variable(Wi.shape[0], f"x{i}")]).T)
            # propogate reference point
            _im_x = Wi * _im_x + bi

            # for all but the last layer (last layer has no ReLU)
            if i < len(self.nn_weights):
                # build indicator vector to encode inequality constraint on xi_hat as dot product
                delta = np.zeros((1, len(_im_x)))
                beta  = np.zeros((1, len(_im_x)))
                for j, _im_x_j in enumerate(_im_x):
                    if _im_x_j[0,0] >= 0:
                        # xi_hat[j] >= 0 iff (1 * xi_hat[j] >= 0)
                        delta[0,j] = 1
                        # constraint xi == xi_hat
                        beta[0,j] = 1
                    else:
                        # xi_hat[j] < 0 iff (-1 * xi_hat[j] > 0)
                        delta[0,j] = -1
                        # constraint xi == 0
                        beta[0,j] = 0

                # continue propogating reference point through f
                _im_x = np.matrix(self.relu(torch.tensor(_im_x)))

                # constraint (xi_hat >= 0 OR xi_hat < 0)
                self.constraints += [delta @ xi_hat >= 0]

                # constraint (xi == xi_hat OR xi == 0)
                self.constraints += [xi_list[-1] == beta @ xi_hat]

        self.constraints += self.constraints_for_k_class_polytope(k=x_class, x=x, compliment=True)

        obj = cp.Minimize(cp.atoms.norm_inf(np.matrix(x) - xi_list[0]))
        problem = cp.Problem(obj, self.constraints)
        problem.solve()

        print(f"f({x}) = {fx} |--> class: {x_class}")
        status = problem.status
        if status == cp.OPTIMAL:
            # TODO: double check whether we need compliment of safety set
            print(f"SUCCESS: ")
            # TODO: compare optimal value to epsilon
            #       if bigger, then we have verified?
            print(f"Opt value = {problem.value}")
        elif status == cp.INFEASIBLE:
            # How to check if this is a false negative? (maybe the relaxations aren't tight enough)
            print(f"COULD NOT verify")
        elif status == cp.UNBOUNDED:
            print(f"Problem is unbounded - {cp.OPTIMAL_INACCURATE}")
        elif status == cp.OPTIMAL_INACCURATE:
            print(f"RUH ROH - {cp.OPTIMAL_INACCURATE}")
            print(f"SUCCESS?")
        elif status == cp.INFEASIBLE_INACCURATE:
            print(f"RUH ROH - {cp.INFEASIBLE_INACCURATE}")
        elif status == cp.UNBOUNDED_INACCURATE:
            print(f"RUH ROH - {cp.UNBOUNDED_INACCURATE}")
        elif status == cp.INFEASIBLE_OR_UNBOUNDED:
            print(f"RUH ROH - {cp.INFEASIBLE_OR_UNBOUNDED}")

        return None


if __name__ == '__main__':

    weights = [
        [[-1, 0],
         [0, 1]],
        [[2, 0],
         [0, -2]],
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
