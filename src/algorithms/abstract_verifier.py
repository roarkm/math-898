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
            self.nn_weights, self.nn_bias_vecs = f.get_weights()

    def __str__(self):
        s = ''
        if not self.f:
            s += "No nn provided."
        else:
            s += f"f:R^{self.nn_weights[0].shape[1]} -> R^{self.nn_weights[-1].shape[0]}"
            s += f"\t{len(self.nn_weights)} layers"
        return s

    def str_constraints(self, layer_id=None, constr_type=None, alg_type=None):
        constrs = self.get_constraints(layer_id=layer_id,
                                       constr_type=constr_type,
                                       alg_type=alg_type)
        if len(constrs) == 0:
            return "NO CONSTRAINTS"
        s = "CONSTRAINTS\n"
        for c in constrs:
            s += f"---------------------------------\n"
            s += str(c)
            s += "\n"
        return s

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
        if len([v for v in self._free_vars if v.name() == var.name()]) != 0:
            s = f"{var.name()} already exists."
            raise Exception(s)
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

    def str_opt_soln(self, var_name=None):
        s = ''
        if var_name:
            opt_var = self.free_vars(var_name=var_name)
            s += f"{opt_var.name()} =\n{opt_var.value}"
            return s
        for opt_var in self.free_vars(var_name=var_name):
            s += f"{opt_var.name()} =\n{opt_var.value}\n"
        return s

    def affine_layer_constraints(self, W, b, layer_id):
        # add constraint for affine transformation
        self.add_free_var(cp.Variable((W.shape[0],1), f"z{layer_id}_hat")) # pre-activation
        self.add_constraint(
            self.free_vars(f"z{layer_id}_hat") == W @ self.free_vars(f"z{layer_id-1}") + b,
            layer_id=layer_id, constr_type='affine', alg_type=self.name)

    def safety_set_constraints(self ,x, verbose=False):
        # Add constraints for safety set
        # Only consider seperating hyperplane for the predicted class of x
        # and the next highest component
        x_class, adversarial_class = self.f.top_two_classes(x)
        n = len(self.nn_weights) # depth of NN
        out_var = self.free_vars(f"z{n}").T
        hplane_constraint = constraints_for_separating_hyperplane(out_var, x_class,
                                                                  adversarial_class,
                                                                  complement=True)
        self.add_constraint(hplane_constraint, layer_id=n,
                            constr_type=f"output",
                            alg_type=self.name)

    def decide_eps_robustness(self, x, eps, verbose=False):
        if self.get_constraints() == []:
            self.network_constraints(x, verbose=verbose)
            c, fv = constraints_for_inf_ball(x, eps, free_vars=self.free_vars('z0'))
            self.add_constraint(c, 0, 'input', self.name)
            self.safety_set_constraints(x, verbose=verbose)

        obj = cp.Minimize(1)
        self.prob = cp.Problem(obj, self.get_constraints())
        self.prob.solve(verbose=verbose,
                        max_iters=10**10,
                        solver=self.solver)
        status = self.prob.status

        if (status == cp.OPTIMAL_INACCURATE) or \
           (status == cp.INFEASIBLE_INACCURATE) or \
           (status == cp.UNBOUNDED_INACCURATE):
            logging.warning("Warning: inaccurate solution.")

        if (status == cp.OPTIMAL) or (status == cp.OPTIMAL_INACCURATE):
            self.counter_example = self.free_vars('z0').value
            return False
        elif status == cp.INFEASIBLE:
            return True
        else:
            raise Exception(status)

    def compute_robustness(self, x, verbose=False):
        if self.get_constraints() == []:
            self.network_constraints(x, verbose=verbose)
            self.safety_set_constraints(x, verbose=verbose)

        obj = cp.Minimize(cp.atoms.norm_inf(np.array(x) - self.free_vars('z0')))
        self.prob = cp.Problem(obj, self.get_constraints())
        self.prob.solve(verbose=verbose,
                        max_iters=10**10,
                        solver=self.solver)
        status = self.prob.status

        if (status == cp.OPTIMAL_INACCURATE) or \
           (status == cp.INFEASIBLE_INACCURATE) or \
           (status == cp.UNBOUNDED_INACCURATE):
            logging.warning("Warning: inaccurate solution.")

        if (status == cp.OPTIMAL) or (status == cp.OPTIMAL_INACCURATE):
            self.counter_example = self.free_vars('z0').value
            return self.prob.value
        elif status == cp.INFEASIBLE:
            return self.prob.value
        else:
            raise Exception(status)

    def verify_counter_example(self, x, counter):
        # assert counter_example is not in safety set!
        assert self.f.class_for_input(x) != self.f.class_for_input(counter)
        # # TODO: assert counter_example is in norm ball!
        # input_constrs = self.get_constraints(constr_typ='input')


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
    return opt_vars @ c >= np.zeros((1,1))


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


if __name__ == '__main__':
    av = AbstractVerifier()
    av.add_constraint('MyConstr1_Affine_ILP',
                      layer_id=1, constr_type='affine', alg_type='ilp')
    print(av.str_constraints(layer_id=1))
