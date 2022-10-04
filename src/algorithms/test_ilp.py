import unittest
import torch
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.abstract_verifier import AbstractVerifier
from src.models.multi_layer import MultiLayerNN, null_map, identity_map

class TestIteratedLinearVerifier(unittest.TestCase):


    def test_identity_network(self):
        # TODO: test combinations of parameterized dimension, depth
        dim = 3
        n_layers = 4
        f = identity_map(dim, n_layers)
        x = torch.rand((1,3))
        assert torch.equal(x, f(x))

        # # TODO: Why does this fail?
        # x = [[1], [1], [1]]
        # for n in range(5):
            # ilp = IteratedLinearVerifier(f)
            # epsilon = 2**(-n)
            # r = ilp.verify_at_point(x=x, eps=epsilon)
            # self.assertTrue(r, debug_failure("\nIdentity map SHOULD NOT be robust", x, epsilon, ilp))
            # del ilp

        x = [[9], [0], [0]]
        for n in range(1, 3):
            ilp = IteratedLinearVerifier(f)
            epsilon = n + 0.3
            r = ilp.verify_at_point(x=x, eps=epsilon)
            self.assertTrue(r, debug_failure("\nIdentity map SHOULD be robust", x, epsilon, ilp))
            del ilp


def debug_failure(err_str, x, epsilon, ilp):
    err_str += f" at {x}.\n"
    err_str += f"Epsilon: {epsilon} < epsilon_hat: {ilp.prob.value}\n"
    err_str += f"Opt Var:\n{ilp.opt_var.value}\n"
    return err_str


if __name__ == '__main__':
    unittest.main()
