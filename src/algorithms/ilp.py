import logging
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from src.models.multi_layer import (MultiLayerNN,
                                    identity_map)
from src.algorithms.abstract_verifier import (AbstractVerifier,
                                              constraints_for_separating_hyperplane)

class IteratedLinearVerifier(AbstractVerifier):

    def __init__(self, f=None):
        super(IteratedLinearVerifier, self).__init__(f)
        logging.basicConfig(format='ILP-%(levelname)s:\n%(message)s', level=logging.INFO)
        self.prob = None

    def build_problem_for_point(self, x, verbose=False):
        # use ILP to verify all x' within eps inf norm of x
        # are classified the same as x'
        # (ie: f(x) == f(x') for all x' in inf-norm ball centered at x
        assert self.f != None, "No NN provided."
        assert self.nn_weights[0].shape[1] == len(x), "x is the wrong shape"

        _im_x = np.reshape(np.array(x), (len(x), 1))
        logging.debug(f"Building for x = \n{_im_x}")

        # z0 is unconstrained since we are using inf-ball in objective function
        # optimization var list starting with z0
        zi_list = [cp.Variable(_im_x.shape, name='z0')]
        self.opt_var = zi_list[0]

        # propogate x layer by layer through f
        # 1) add constraints for affine transforms
        # 2) add constraints for ReLU activation pattern
        for i in range(0, len(self.nn_weights)):
            Wi = self.nn_weights[i]
            _bi = self.nn_bias_vecs[i]
            bi = np.reshape(_bi, (_bi.shape[0], 1))

            # add constraint for affine transformation
            zi_list.append(cp.Variable((Wi.shape[0],1), f"z{i+1}_hat"))
            logging.debug(zi_list[-1] == Wi @ zi_list[-2] + bi)
            self.constraints += [zi_list[-1] == Wi @ zi_list[-2] + bi]

            # propogate reference point
            _im_x = Wi @ _im_x + bi
            logging.debug(f"After layer {i} affine T - shape={_im_x.shape}")
            logging.debug(_im_x)

            # for all but the last layer
            # TODO: Do not assume last layer has no ReLU (how to verify?)
            if i < len(self.nn_weights):
                # add constraints for ReLU activation pattern
                zi_list.append(cp.Variable((Wi.shape[0],1), f"z{i+1}"))
                # build indicator vector to encode inequality
                # constraint on zi_hat as dot product
                delta = np.zeros((1, len(_im_x)))
                beta  = np.zeros((1, len(_im_x)))
                for j, _im_x_j in enumerate(_im_x):
                    logging.debug(f"layer-{i}, _im_x[{j}] = {_im_x_j[0]}")
                    # continue
                    if _im_x_j[0] >= 0:
                        # zi_hat[j] >= 0 iff (1 * zi_hat[j] >= 0)
                        delta[0,j] = 1
                        # constraint zi == zi_hat
                        beta[0,j] = 1
                    else:
                        # zi_hat[j] < 0 iff (-1 * zi_hat[j] > 0)
                        delta[0,j] = -1
                        # constraint zi == 0
                        beta[0,j] = 0

                # constraint (zi_hat >= 0 OR zi_hat < 0)
                self.constraints += [delta @ zi_list[-2] >= 0]

                # constraint (xi == zi_hat OR xi == 0)
                self.constraints += [zi_list[-1] == beta @ zi_list[-2]]

                # continue propogating reference point through f
                logging.debug(f"Before relu at layer {i}: shape={_im_x.shape}")
                logging.debug(_im_x)
                _im_x = np.array(self.relu(torch.tensor(_im_x)))
                logging.debug(f"After relu at layer {i}: shape={_im_x.shape}")
                logging.debug(_im_x)

        # Add constraints for safety set
        # Only consider seperating hyperplane for the predicted class of x
        # and the next highest component
        fx = self.f(torch.tensor(x).T.float()).detach().numpy()
        assert np.array_equal(_im_x, fx.T) == True # sanity check

        _class_order = np.argsort(fx)[0]
        x_class = _class_order[-1]           # index of component with largest value
        adversarial_class = _class_order[-2] # index of component with second largest value
        logging.debug(f"f({x}) = {fx} => Want z_{adversarial_class} > z_{x_class}")

        self.constraints += constraints_for_separating_hyperplane(zi_list[-1].T, x_class,
                                                                  adversarial_class,
                                                                  complement=True)

        obj = cp.Minimize(cp.atoms.norm_inf(np.array(x) - zi_list[0]))
        self.prob = cp.Problem(obj, self.constraints)
        logging.debug("Constraints")
        logging.debug(self.str_constraints())

    def robustness_at_point(self, x, verbose=False):
        self.prob.solve(verbose=verbose)
        status = self.prob.status
        if status == cp.OPTIMAL:
            return self.prob.value
        elif status == cp.OPTIMAL_INACCURATE:
            logging.warning("Warning: inaccurate solution.")
            return self.prob.value
        else:
            raise Exception(status)

    def verify_at_point(self, x=[[9], [-9]], eps=0.5, verbose=False):
        self.build_problem_for_point(x=x, verbose=verbose)
        try:
            eps_hat = self.robustness_at_point(x, verbose=verbose)
            if eps_hat < eps:
                return False
            return True
        except Exception as err:
            logging.critical(err)



if __name__ == '__main__':
    f = identity_map(2,2)
    ilp = IteratedLinearVerifier(f)
    eps = 0.5
    x = [[9], [0]]
    is_robust = ilp.verify_at_point(x=x, eps=eps)

    # TODO: need some tolerance built in?
    if is_robust:
        im_adv = f(torch.tensor(ilp.opt_var.value).T.float()).detach().numpy()
        fx = f(torch.tensor(x).T.float()).detach().numpy()
        logging.info(f"Identity map is ({eps})-robust at x={x}. epsilon-hat={ilp.prob.value} > epsilon={eps}")
        logging.info(f"Best z0:\n{ilp.opt_var.value}")
    else:
        logging.debug(ilp.str_constraints())
        logging.info(f"Best z0: {ilp.opt_var.value}")
        logging.info(f"Identity map is NOT ({eps})-robust at x={x}: epsilon_hat: {ilp.prob.value}")
