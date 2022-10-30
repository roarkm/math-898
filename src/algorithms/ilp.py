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
        logging.basicConfig(format='ILP-%(levelname)s:\n%(message)s', level=logging.INFO)
        self.prob = None

    def constraints_for_affine_layer(self, W, b, layer_id):
        # add constraint for affine transformation
        self.add_free_var(cp.Variable((W.shape[0],1), f"z{layer_id}_hat")) # pre-activation
        self.add_constraint(
            self.free_vars(f"z{layer_id}_hat") == W @ self.free_vars(f"z{layer_id-1}") + b,
            layer_id=layer_id, constr_type='affine', alg_type='ilp')

        logging.debug(self.free_vars(names_only=True))
        logging.debug(self.str_constraints(layer_id=layer_id, constr_type='affine', alg_type='ilp'))

    def constraints_for_point(self, x, verbose=False):
        # use ILP to verify all x' within eps inf norm of x
        # are classified the same as x'
        # (ie: f(x) == f(x') for all x' in inf-norm ball centered at x
        assert self.f != None, "No NN provided."
        assert self.nn_weights[0].shape[1] == len(x), "x is the wrong shape"

        _im_x = np.reshape(np.array(x), (len(x), 1))

        # z0 is unconstrained since we are using inf-ball in objective function
        # optimization var list starting with z0
        # TODO: optionally use constraints for region of interest
        self.add_free_var(cp.Variable(_im_x.shape, name='z0'))

        # propogate x layer by layer through f
        # 1) add constraints for affine transforms
        # 2) add constraints for ReLU activation pattern
        for i in range(1, len(self.nn_weights)+1):
            Wi = self.nn_weights[i-1]
            _bi = self.nn_bias_vecs[i-1]
            bi = np.reshape(_bi, (_bi.shape[0], 1))

            self.constraints_for_affine_layer(Wi, bi, i)

            # propogate reference point
            _im_x = Wi @ _im_x + bi
            logging.debug(f"After layer {i} affine T - shape={_im_x.shape}")
            logging.debug(_im_x)

            # add constraints for ReLU activation pattern
            self.add_free_var(cp.Variable((Wi.shape[0],1), f"z{i}")) # post-activation

            # build indicator vector to encode inequality
            # constraint on zi_hat as dot product
            delta = np.zeros((1, len(_im_x)))
            beta  = np.zeros((1, len(_im_x)))
            for j, _im_x_j in enumerate(_im_x):
                if _im_x_j[0] >= 0:
                    delta[0,j] = 1 # zi_hat[j] >= 0 iff (1 * zi_hat[j] >= 0)
                    beta[0,j] = 1  # constraint zi == zi_hat
                else:
                    delta[0,j] = -1 # zi_hat[j] < 0 iff (-1 * zi_hat[j] > 0)
                    beta[0,j] = 0   # constraint zi == 0

            # constraint (zi_hat >= 0 OR zi_hat < 0)
            self.add_constraint(delta @ self.free_vars(f"z{i}_hat") >= 0,
                                layer_id=i,
                                constr_type='relu_1',
                                alg_type='ilp')

            # constraint (xi == zi_hat OR xi == 0)
            self.add_constraint(self.free_vars(f"z{i}") == beta @ self.free_vars(f"z{i}_hat"),
                                layer_id=i,
                                constr_type='relu_2',
                                alg_type='ilp')
            logging.debug(self.str_constraints(layer_id=i,
                                               constr_type='relu_2',
                                               alg_type='ilp'))

            # continue propogating reference point through f
            logging.debug(f"Before relu at layer {i}: shape={_im_x.shape}")
            logging.debug(_im_x)
            _im_x = np.array(self.relu(torch.tensor(_im_x)))

        # Add constraints for safety set
        # Only consider seperating hyperplane for the predicted class of x
        # and the next highest component
        fx = self.f(torch.tensor(x).T.float()).detach().numpy()
        assert np.array_equal(_im_x, fx.T) == True # sanity check

        _class_order = np.argsort(fx)[0]
        x_class = _class_order[-1]           # index of component with largest value
        adversarial_class = _class_order[-2] # index of component with second largest value
        logging.debug(f"f({x}) = {fx} => Want z_{adversarial_class} > z_{x_class}")

        n = len(self.nn_weights) # depth of NN
        out_var = self.free_vars(f"z{n}").T
        hplane_constraint = constraints_for_separating_hyperplane(out_var, x_class,
                                                                  adversarial_class,
                                                                  complement=True)
        self.add_constraint(hplane_constraint, layer_id=n,
                            constr_type=f"safety_set",
                            alg_type='ilp')
        return self.get_constraints()

    def problem_for_point(self, x, verbose=False):
        if self.get_constraints() == []:
            self.constraints_for_point(x, verbose=verbose)

        obj = cp.Minimize(cp.atoms.norm_inf(np.array(x) - self.free_vars('z0')))
        self.prob = cp.Problem(obj, self.get_constraints())
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

    def verify_at_point(self, x=[[9], [-9]], eps=0.5, verbose=False, tol=10**(-4)):
        self.problem_for_point(x=x, verbose=verbose)
        try:
            eps_hat = self.robustness_at_point(x, verbose=verbose)
            if eps_hat < eps - tol:
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

    if is_robust:
        im_adv = f(torch.tensor(ilp.free_vars('z0').value).T.float()).detach().numpy()
        fx = f(torch.tensor(x).T.float()).detach().numpy()
        logging.info(f"Identity map is ({eps})-robust at x={x}. epsilon-hat={ilp.prob.value} > epsilon={eps}")
        logging.info(f"Best z0:\n{ilp.free_vars('z0').value}")
    else:
        logging.debug(ilp.str_constraints())
        logging.info(f"Best z0: {ilp.free_vars('z0').value}")
        logging.info(f"Identity map is NOT ({eps})-robust at x={x}: epsilon_hat: {ilp.prob.value}")
