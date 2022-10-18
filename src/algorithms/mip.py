import logging
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from src.models.multi_layer import (MultiLayerNN,
                                    identity_map)
from src.algorithms.abstract_verifier import (AbstractVerifier,
                                              constraints_for_separating_hyperplane)

# TODO: log the branch and bound nodes that the optimizer builds for the MIP

class MIPVerifier(AbstractVerifier):

    def __init__(self, f=None):
        super(MIPVerifier, self).__init__(f)
        self.name = 'NSVerify'
        logging.basicConfig(format='ILP-%(levelname)s:\n%(message)s', level=logging.INFO)
        self.prob = None

    def constraints_for_network(self, verbose=False):
        assert self.f != None, "No NN provided."
        assert self.nn_weights[0].shape[1] == len(x), "x is the wrong shape"

        M = 10**10
        logging.debug(f"Setting M = {M}")

        # z0 is unconstrained since we are using inf-ball in objective function
        # optimization var list starting with z0
        self.add_free_var(cp.Variable((self.nn_weights[0].shape[1],1), name='z0'))

        # propogate x layer by layer through f
        # 1) add constraints for affine transforms
        # 2) add constraints for ReLU activation pattern
        for i in range(1, len(self.nn_weights)+1):
            Wi = self.nn_weights[i-1]
            _bi = self.nn_bias_vecs[i-1]
            bi = np.reshape(_bi, (_bi.shape[0], 1))

            # add constraint for affine transformation
            self.add_free_var(cp.Variable((Wi.shape[0],1), f"z{i}_hat")) # pre-activation
            # constrain pre-activation var with affine transformation
            self.constraints += [self.free_vars(f"z{i}_hat") == Wi @ self.free_vars(f"z{i-1}") + bi]
            logging.debug(self.constraints[-1])

            # for all but the last layer
            # TODO: Do not assume last layer has no ReLU (how to verify?)
            if i < len(self.nn_weights):
                self.add_free_var(cp.Variable((Wi.shape[0],1), f"z{i}_hat")) # pre-activation
                self.add_free_var(cp.Variable((Wi.shape[0],1), f"z{i}")) # post-activation
                self.add_free_var(cp.Variable((Wi.shape[0],1), f"d{i}", integer=True))

                # d_ij in {0,1}
                self.constraints += [self.free_vars(f"d{i}") >= 0]
                self.constraints += [self.free_vars(f"d{i}") <= 1]

                # z_ij >= z_ij_hat
                self.constraints += [self.free_vars(f"z{i}") >= self.free_vars(f"z{i}_hat")]

                # z_ij_hat < 0 implies d_ij == 0    iff    -z_ij <= M * (1 - d_ij)
                self.constraints += [-1*self.free_vars(f"z{i}_hat") <= M*(np.ones((Wi.shape[0], 1))-self.free_vars(f"d{i}"))]

                # z_ij_hat > 0 implies d_ij == 1    iff    z_ij <= M * d_ij
                self.constraints += [self.free_vars(f"z{i}_hat") <= M * self.free_vars(f"d{i}")]

        # Add constraints for safety set
        # Only consider seperating hyperplane for the predicted class of x
        # and the next highest component
        fx = self.f(torch.tensor(x).T.float()).detach().numpy()

        _class_order = np.argsort(fx)[0]
        x_class = _class_order[-1]           # index of component with largest value
        adversarial_class = _class_order[-2] # index of component with second largest value
        logging.debug(f"f({x}) = {fx} => Want z_{adversarial_class} > z_{x_class}")
        n = len(self.nn_weights)
        # out_var = self.free_vars(f"z{n}").T      # TODO: if last layer ReLU
        out_var = self.free_vars(f"z{n}_hat").T

        self.constraints += constraints_for_separating_hyperplane(out_var, x_class,
                                                                  adversarial_class,
                                                                  complement=True)
        return self.constraints

    def build_problem_for_point(self, x, verbose=False):
        if self.constraints == []:
            self.constraints_for_network(verbose=verbose)

        obj = cp.Minimize(cp.atoms.norm_inf(np.array(x) - self.free_vars('z0')))
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
    mip = MIPVerifier(f)
    eps = 0.5
    x = [[9], [0]]
    is_robust = mip.verify_at_point(x, eps)

    if is_robust:
        im_adv = f(torch.tensor(mip.free_vars('z0').value).T.float()).detach().numpy()
        fx = f(torch.tensor(x).T.float()).detach().numpy()
        logging.info(f"Identity map is ({eps})-robust at x={x}. epsilon-hat={mip.prob.value} > epsilon={eps}")
        logging.info(f"Best z0:\n{mip.free_vars('z0').value}")
    else:
        logging.debug(mip.str_constraints())
        logging.info(f"Best z0: {mip.free_vars('z0').value}")
        logging.info(f"Identity map is NOT ({eps})-robust at x={x}: epsilon_hat: {mip.prob.value}")
