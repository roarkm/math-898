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
        # self.solver = cp.ECOS_BB # has bugs
        self.solver = cp.GLPK_MI
        self.M = M
        logging.basicConfig(format='MIP-%(levelname)s:\n%(message)s', level=logging.DEBUG)
        self.prob = None

    def relu_constraints(self, Wi, layer_id):
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

    def network_constraints(self, _=None, verbose=False):
        assert self.f != None, "No NN provided."

        # optimization var list starting with z0
        self.add_free_var(cp.Variable((self.nn_weights[0].shape[1],1), name='z0'))

        for i in range(1, len(self.nn_weights)+1):
            # get layer weights and bias vec
            Wi = self.nn_weights[i-1]
            _bi = self.nn_bias_vecs[i-1]
            bi = np.reshape(_bi, (_bi.shape[0], 1))

            # add affine constraints
            self.affine_layer_constraints(Wi, bi, i)
            # add relu constraints
            self.relu_constraints(Wi, i)

        return self.get_constraints()


def quick_test_eps_robustness():
    f = identity_map(2,2)
    mip = MIPVerifier(f)
    eps = 1
    x = [[9], [3]]
    e_robust = mip.decide_eps_robustness(x, eps, verbose=True)

    logging.debug(mip.str_constraints())

    x_class = f.class_for_input(x)
    print(f"f({x}) = class {x_class+1}")
    print(f"{f.name} is ({eps})-robust at {x}?  {e_robust}")
    if not e_robust:
        print(mip.str_opt_soln('z0'))
        ce = mip.counter_example
        print(f"Counterexample: f({ce}) = class {f.class_for_input(ce)+1}")


def quick_test_pointwise_robustness():
    f = identity_map(2,2)
    mip = MIPVerifier(f)
    x = [[2.01], [1]]
    eps_hat = mip.compute_robustness(x)

    logging.debug(mip.str_constraints())

    print(f"Pointwise robusntess of {f.name} at {x} is {eps_hat}.")
    print(f"Nearest adversarial example is \n{mip.counter_example}.")

if __name__ == '__main__':
    quick_test_eps_robustness()
    quick_test_pointwise_robustness()
