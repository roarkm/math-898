import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import logging

class AbstractVerifier():

    def __init__(self, f=None):
        self.f = f
        self.name = 'AbstractVerifier'
        self._constraints = {}
        self._free_vars = []
        self.relu = nn.ReLU()
        if f:
            self.nn_weights, self.nn_bias_vecs = weights_from_nn(self.f)

    def constraints_for_affine_layer(self, W, b, layer_id):
        # add constraint for affine transformation
        self.add_free_var(cp.Variable((W.shape[0],1), f"z{layer_id}_hat")) # pre-activation
        self.add_constraint(
            self.free_vars(f"z{layer_id}_hat") == W @ self.free_vars(f"z{layer_id-1}") + b,
            layer_id=layer_id, constr_type='affine', alg_type=self.name)

        logging.debug(self.free_vars(names_only=True))
        logging.debug(self.str_constraints(layer_id=layer_id, constr_type='affine', alg_type='mip'))


    def add_constraint(self, constr, layer_id, constr_type, alg_type):
        self._constraints[(layer_id, constr_type, alg_type)] = constr

    def get_constraints(self, layer_id=None, constr_type=None, alg_type=None):
        if len(self._constraints) == 0:
            return []
        _c = self._constraints
        if layer_id is not None:
            _c = {k:v for (k,v) in _c.items() if k[0] == layer_id}
        if constr_type is not None:
            _c = {k:v for (k,v) in _c.items() if k[1] == constr_type}
        if alg_type is not None:
            _c = {k:v for (k,v) in _c.items() if k[2] == alg_type}
        return list(_c.values())

    def add_free_var(self, var):
        self._free_vars.append(var)

    def free_vars(self, var_name=None, names_only=False):
        if names_only:
            if self._free_vars == []:
                return []
            return [v.name() for v in self._free_vars]
        if var_name is None:
            return self._free_vars
        try:
            return [v for v in self._free_vars if v.name() == var_name][0]
        except IndexError:
            s = f"{var_name} not in free vars."
            raise Exception(s)

    def str_constraints(self, layer_id=None, constr_type=None, alg_type=None):
        return str_constraints(self.get_constraints(layer_id=layer_id,
                                                    constr_type=constr_type,
                                                    alg_type=alg_type))

    def __str__(self):
        s = ''
        if not self.f:
            s += "No nn provided."
        else:
            s += f"f:R^{self.nn_weights[0].shape[1]} -> R^{self.nn_weights[-1].shape[0]}"
            s += f"\t{len(self.nn_weights)} layers"
        return s

    def sep_hplane_for_advclass(self, x, complement=False):
        fx = self.f(torch.tensor(np.array(x)).T.float()).detach().numpy()
        class_order = np.argsort(fx)[0]
        # index of component with largest value
        x_class = class_order[-1]
        # index of component with second largest value
        adversarial_class = class_order[-2]
        return _vector_for_separating_hyperplane(large_index=x_class,
                                                 small_index=adversarial_class,
                                                 n=fx.shape[1],
                                                 complement=complement)

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

    def problem_for_point(self, x, verbose=False):
        if self.get_constraints() == []:
            self.constraints_for_point(x, verbose=verbose)
        # Currently using max-disturbance
        # TODO: optionally constrain input region instead of max-disturbance
        obj = cp.Minimize(cp.atoms.norm_inf(np.array(x) - self.free_vars('z0')))
        self.prob = cp.Problem(obj, self.get_constraints())
        logging.debug("Constraints")
        logging.debug(self.str_constraints())

    def verify_at_point(self, x=[[9], [-9]], eps=0.5, verbose=False, tol=10**(-4)):
        self.problem_for_point(x=x, verbose=verbose)
        try:
            eps_hat = self.robustness_at_point(x, verbose=verbose)
            if eps_hat < eps - tol:
                return False
            return True
        except Exception as err:
            logging.critical(err)


def _vector_for_separating_hyperplane(large_index, small_index, n, complement=False):
    # create vector c for a seperating hyperplane
    #    {x | c dot x >= 0 and x,c in R^n} = {x | x_large_index > x_small_index }
    c = np.zeros((1, n))
    c[0][large_index] = 1
    c[0][small_index] = -1
    if complement:
        return -1 * c.T
    return c.T


def constraints_for_separating_hyperplane(opt_vars, large_index, small_index,
                                          verbose=False, complement=False):
    # create constraint for a seperating hyperplane in R^n
    # where n = dim(opt_vars)
    assert large_index <= opt_vars.shape[1]
    assert small_index <= opt_vars.shape[1]

    c = _vector_for_separating_hyperplane(large_index, small_index,
                                          opt_vars.shape[1],
                                          complement=complement)
    return c @ opt_vars >= np.zeros((1,1))


def mat_for_k_class_polytope(k, dim, complement=False):
    assert complement == False, "Complement option not yet supported"
    if dim == 1:
        return np.zeros((1,1)), np.zeros((1,1))

    n_rows = dim-1
    A = np.zeros((n_rows, dim))
    A[:,k] = 1

    row = 0
    for j in range(0, dim):
        if j == k:
            continue
        A[row][j] = -1
        row += 1
    b = np.zeros((n_rows, 1))
    return (A, b)


def constraints_for_k_class_polytope(k, x, verbose=False, complement=False):
    # build a polytope constraint matrix corresponding to the output region
    # of the k-th class
    # ie - the polytope corresponding to the region of R^dim where
    # the k-th component is greater than all other components
    # returns a list containing the inequality constraints in matrix form
    z = cp.bmat([cp.Variable(len(x), name='z')]).T
    A, b = mat_for_k_class_polytope(k, len(x), complement)
    return A @ z >= b


def constraints_for_inf_ball(center, eps, free_vars=None, free_vars_name=None):
    # create a list containing the constraints for an inf-ball of radius eps
    if free_vars is not None:
        assert free_vars.shape[0] == len(center)
    else:
        if free_vars_name == None:
            free_vars_name = 'z0'
        free_vars = cp.Variable((len(center),1), name=free_vars_name)

    A_lt = np.zeros((2, len(center)))
    b_vec = np.zeros((2, 1))

    # top constraint
    A_lt[0][0] = 1
    b_vec[0][0] = center[0][0] + eps
    # bottom constraint
    A_lt[1][0] = -1
    b_vec[1][0] = -1 * (center[0][0] - eps)

    if len(center) > 1:
        for i in range(1, len(center)):
            # top constraint
            row = np.zeros((1, len(center)))
            row[0][i] = 1
            A_lt = np.vstack([A_lt, row])
            _a = np.matrix([[ float(center[i][0]) + eps ]])
            b_vec = np.vstack([b_vec, _a])

            # bottom constraint
            row = np.zeros((1, len(center)))
            row[0][i] = -1
            A_lt = np.vstack([A_lt, row])
            _a = np.matrix([[ -1 * (float(center[i][0]) - eps) ]])
            b_vec = np.vstack([b_vec, _a])
    return A_lt @ free_vars <= b_vec, free_vars


def str_constraints(constraints):
    if len(constraints) == 0:
        return "No Constraints."
    s = ""
    for c in constraints:
        s += f"constraint:\n"
        s += str(c)
        s += "\n"
    return s


def weights_from_nn(f):
    """
    :type  : MultiLayerNN
    :rtype : tuple(list[np.array], list[np.array])
    """
    # only handles 'flat' ffnn's (for now)
    # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    weights, bias_vecs = [], []
    for i, l in enumerate(f.layers):
        if isinstance(l, nn.modules.linear.Linear):
            weights.append(l.weight.data.numpy())
            bias_vecs.append(l.bias.data.numpy())
        else:
            assert isinstance(l, nn.modules.activation.ReLU)
    assert len(weights) >= 2, "Only supporting more than two layers"
    return weights, bias_vecs


if __name__ == '__main__':
    av = AbstractVerifier()
    av.add_constraint('MyConstr0_ReLU_ILP',
                      layer_id=0, constr_type='relu', alg_type='ilp')
    av.add_constraint('MyConstr0_ReLU_MIP',
                      layer_id=0, constr_type='relu', alg_type='mip')
    av.add_constraint('MyConstr0_Affine_ILP',
                      layer_id=0, constr_type='affine', alg_type='ilp')
    av.add_constraint('MyConstr1_Affine_ILP',
                      layer_id=1, constr_type='affine', alg_type='ilp')
    print(av.str_constraints(layer_id=0))
