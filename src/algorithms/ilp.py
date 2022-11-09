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
        self.name = 'ILP'
        logging.basicConfig(format='ILP-%(levelname)s:\n%(message)s', level=logging.DEBUG)
        self.prob = None

    def relu_constraints(self, im_x, layer_id):
        # add constraints for ReLU activation pattern
        self.add_free_var(cp.Variable((im_x.shape[0],1), f"z{layer_id}")) # post-activation

        # build indicator vector to encode inequality
        # constraint on zi_hat as dot product
        # delta = np.zeros((len(im_x), len(im_x)))
        delta = np.zeros((1, len(im_x)))
        for j, _im_x_j in enumerate(im_x):
            if _im_x_j[0] > 0:
                delta[0,j] = 1

        self.add_constraint(
            self.free_vars(f"z{layer_id}") == np.diagflat(delta) @ self.free_vars(f"z{layer_id}_hat"),
            layer_id=layer_id,
            constr_type='relu_1',
            alg_type='ilp')

        _delta = 2*delta - np.ones((1,len(im_x)))
        self.add_constraint(
                np.diagflat(_delta) @ self.free_vars(f"z{layer_id}_hat") >= 0,
                layer_id=layer_id,
                constr_type='relu_2',
                alg_type='ilp')


    def network_constraints(self, x, verbose=False):
        # build constraints for the network at x
        assert self.f != None, "No NN provided."
        assert self.nn_weights[0].shape[1] == len(x), "x is the wrong shape"

        _im_x = np.reshape(np.array(x), (len(x), 1))

        # optimization var list starting with z0
        # z0 to be constrained elsewhere for sat problem
        self.add_free_var(cp.Variable(_im_x.shape, name='z0'))

        for i in range(1, len(self.nn_weights)+1):
            # Get layer weights and bias vec
            Wi = self.nn_weights[i-1]
            _bi = self.nn_bias_vecs[i-1]
            bi = np.reshape(_bi, (_bi.shape[0], 1))

            # propogate reference point
            _im_x = Wi @ _im_x + bi

            # add affine constraints
            self.affine_layer_constraints(Wi, bi, i)
            # add relu constraints
            self.relu_constraints(_im_x, i)

            # continue propogating reference point through f
            _im_x = np.array(self.relu(torch.tensor(_im_x)))

        fx = self.f(torch.tensor(x).T.float()).detach().numpy()
        assert np.array_equal(_im_x.astype('float32'), fx.T.astype('float32'))
        return self.get_constraints()



def quick_test_eps_robustness():
    f = identity_map(2,2)
    ilp = IteratedLinearVerifier(f)
    eps = 8
    x = [[9], [1.1]]
    e_robust = ilp.decide_eps_robustness(x, eps, verbose=True)

    # logging.debug(ilp.str_constraints())

    x_class = f.class_for_input(x)
    print(f"f({x}) = class {x_class+1}")
    print(f"{f.name} is ({eps})-robust at {x}?  {e_robust}")
    if not e_robust:
        print(ilp.str_opt_soln('z0'))
        ce = ilp.counter_example
        print(f"Counterexample: f({ce}) = class {f.class_for_input(ce)+1}")


def quick_test_pointwise_robustness():
    f = identity_map(2,2)
    ilp = IteratedLinearVerifier(f)
    x = [[2.01], [1]]
    eps_hat = ilp.compute_robustness(x)
    print(f"Pointwise robusntess of {f.name} at {x} is {eps_hat}.")
    print(f"Nearest adversarial example is \n{ilp.counter_example}.")

if __name__ == '__main__':
    quick_test_eps_robustness()
    quick_test_pointwise_robustness()
