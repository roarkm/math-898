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

    def __init__(self, f=None, M=10**3):
        super(MIPVerifier, self).__init__(f)
        self.name = 'NSVerify'
        self.M = M
        logging.basicConfig(format='ILP-%(levelname)s:\n%(message)s', level=logging.INFO)
        self.prob = None

    def constraints_for_relu(self, Wi, layer_id):
        self.add_free_var(cp.Variable((Wi.shape[0],1), f"z{layer_id}_hat")) # pre-activation
        self.add_free_var(cp.Variable((Wi.shape[0],1), f"z{layer_id}")) # post-activation
        self.add_free_var(cp.Variable((Wi.shape[0],1), f"d{layer_id}", integer=True))

        # d_ij in {0,1}
        self.add_constraint(self.free_vars(f"d{layer_id}") >= 0,
                            layer_id=layer_id,
                            constr_type='int_GT_0',
                            alg_type='mip')
        self.add_constraint(self.free_vars(f"d{layer_id}") <= 1,
                            layer_id=layer_id,
                            constr_type='int_LT_0',
                            alg_type='mip')

        # z_ij >= z_ij_hat
        self.add_constraint(self.free_vars(f"z{layer_id}") >= self.free_vars(f"z{layer_id}_hat"),
                            layer_id=layer_id,
                            constr_type='relu_1',
                            alg_type='mip')

        # z_ij_hat < 0 implies d_ij == 0    iff    -z_ij <= M * (1 - d_ij)
        _c = -1*self.free_vars(f"z{layer_id}_hat") <= self.M * (np.ones((Wi.shape[0], 1))-self.free_vars(f"d{layer_id}"))
        self.add_constraint(_c,
                            layer_id=layer_id,
                            constr_type='relu_2',
                            alg_type='mip')

        # z_ij_hat > 0 implies d_ij == 1    iff    z_ij <= M * d_ij
        _c = self.free_vars(f"z{layer_id}_hat") <= self.M * self.free_vars(f"d{layer_id}")
        self.add_constraint(_c,
                            layer_id=layer_id,
                            constr_type='relu_3',
                            alg_type='mip')
        logging.debug(self.str_constraints(layer_id=layer_id))

    def constraints_for_point(self, _=None, verbose=False):
        assert self.f != None, "No NN provided."
        assert self.nn_weights[0].shape[1] == len(x), "x is the wrong shape"

        logging.debug(f"Setting M = {self.M}")

        # z0 is unconstrained since we are using inf-ball in objective function
        # optimization var list starting with z0
        self.add_free_var(cp.Variable((self.nn_weights[0].shape[1],1), name='z0'))

        for i in range(1, len(self.nn_weights)+1):
            # get layer weights and bias vec
            Wi = self.nn_weights[i-1]
            _bi = self.nn_bias_vecs[i-1]
            bi = np.reshape(_bi, (_bi.shape[0], 1))

            # add affine constraints
            self.constraints_for_affine_layer(Wi, bi, i)
            # add relu constraints
            self.constraints_for_relu(Wi, i)

        # Add constraints for safety set
        # Only consider seperating hyperplane for the predicted class of x
        # and the next highest component
        fx = self.f(torch.tensor(x).T.float()).detach().numpy()

        _class_order = np.argsort(fx)[0]
        x_class = _class_order[-1]           # index of component with largest value
        adversarial_class = _class_order[-2] # index of component with second largest value

        n = len(self.nn_weights) # depth of NN
        out_var = self.free_vars(f"z{n}").T
        _c = constraints_for_separating_hyperplane(out_var, x_class,
                                                   adversarial_class,
                                                   complement=True)
        self.add_constraint(_c, layer_id=n,
                            constr_type='safety_set',
                            alg_type='ilp')
        return self.get_constraints()


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
