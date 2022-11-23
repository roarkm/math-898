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
        self.params = {'M': self.M}
        logging.basicConfig(format='MIP-%(levelname)s:\n%(message)s',
                            level=logging.INFO)
        self.prob = None

    def relu_constraints(self, Wi, layer_id):
        self.add_free_var(cp.Variable((Wi.shape[0], 1),
                                      f"z{layer_id}"))
        self.add_free_var(cp.Variable((Wi.shape[0], 1),
                                      f"d{layer_id}", integer=True))

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
        c = (self.free_vars(f"z{layer_id}")
             >= self.free_vars(f"z{layer_id}_hat"))
        self.add_constraint(c, layer_id=layer_id,
                            constr_type='relu_1',
                            alg_type='mip')
        # z_ij >= 0
        c = (self.free_vars(f"z{layer_id}")
             >= np.zeros((Wi.shape[0], 1)))
        self.add_constraint(c, layer_id=layer_id,
                            constr_type='relu_2',
                            alg_type='mip')

        # z_ij <= z_ij_hat + M(1 - d_ij)
        c = (self.free_vars(f"z{layer_id}")
             <= (self.free_vars(f"z{layer_id}_hat")
                 + cp.multiply(self.M,
                               (np.ones((Wi.shape[0], 1))
                                - self.free_vars(f"d{layer_id}")))))
        self.add_constraint(c, layer_id=layer_id,
                            constr_type='relu_3',
                            alg_type='mip')

        #  z_ij <= M * d_ij
        c = (self.free_vars(f"z{layer_id}")
             <= cp.multiply(self.M, self.free_vars(f"d{layer_id}")))
        self.add_constraint(c,
                            layer_id=layer_id,
                            constr_type='relu_4',
                            alg_type='mip')

    def network_constraints(self, _=None, verbose=False):
        assert self.f is not None, "No NN provided."

        # optimization var list starting with z0
        self.add_free_var(cp.Variable((self.nn_weights[0].shape[1], 1),
                                      name='z0'))

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
    f = identity_map(2, 2)
    mip = MIPVerifier(f)
    eps = 1
    # x = [[4], [4.01]]
    x = [[9], [1]]
    e_robust = mip.decide_eps_robustness(x, eps)

    # print(mip.str_constraints())
    # print(mip.str_opt_soln())

    x_class = f.class_for_input(x)
    print(f"f({x}) = class {x_class}")
    print(f"{f.name} is ({eps})-robust at {x}?  {e_robust}")
    if not e_robust:
        ce = mip.counter_example
        print(f"Counterexample: f({ce}) = class {f.class_for_input(ce)}")


def quick_test_pointwise_robustness():
    f = identity_map(2, 2)
    mip = MIPVerifier(f)
    x = [[2.01], [1]]
    eps_hat = mip.compute_robustness(x)

    # print(mip.str_constraints())

    print(f"Pointwise robusntess of {f.name} at {x} is {eps_hat}.")
    print(f"Nearest adversarial example is \n{mip.counter_example}.")


if __name__ == '__main__':
    quick_test_eps_robustness()
    # quick_test_pointwise_robustness()
