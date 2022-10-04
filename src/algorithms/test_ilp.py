import unittest
import torch
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.abstract_verifier import AbstractVerifier
from src.models.multi_layer import MultiLayerNN

class TestIteratedLinearVerifier(unittest.TestCase):

    def null_map_network(self):
        weights = [
            [[1, 0],
             [0, 1]],
            [[0, 0],
             [0, 0]],
        ]
        bias_vecs =[
            (1,1),
            (0,0),
        ]
        f = MultiLayerNN(weights, bias_vecs)
        x = torch.rand((1,2))
        y = torch.zeros((1,2))
        assert torch.equal(y, f(x))
        # this fails if called after the decision boundary test!
        x = [[9], [0]]
        for n in range(1, 3):
            ilp = IteratedLinearVerifier(f)
            epsilon = n + 0.3
            self.assertTrue(ilp.verify_at_point(x=x, eps=epsilon),
                            f"Null map should always be {epsilon}-robust at {x}.")
            del ilp


    def test_identity_network(self):
        # TODO: test combinations of parameterized dimension, depth
        weights = [
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
        ]
        bias_vecs =[
            (0,0,0),
            (0,0,0),
            (0,0,0),
            (0,0,0),
        ]
        f = MultiLayerNN(weights, bias_vecs)
        x = torch.rand((1,3))
        assert torch.equal(x, f(x))

        # this fails if called after the decision boundary test!
        x = [[9], [0], [0]]
        for n in range(1, 3):
            # SCS hangs for eps >= 4
            ilp = IteratedLinearVerifier(f)
            epsilon = n + 0.3
            r = ilp.verify_at_point(x=x, eps=epsilon)
            self.assertTrue(r, debug_failure("\nIdentity map SHOULD be robust", x, epsilon, ilp))
            del ilp

        # TODO: Bug! if we call this code block before the above, said test fails!
        x = [[1], [1], [1]]
        for n in range(5):
            ilp = IteratedLinearVerifier(f)
            epsilon = 2**(-n)
            r = ilp.verify_at_point(x=x, eps=epsilon)
            self.assertFalse(r, debug_failure("\nIdentity map SHOULD NOT be robust", x, epsilon, ilp))
            del ilp


def debug_failure(err_str, x, epsilon, ilp):
    err_str += f" at {x}.\n"
    err_str += f"Epsilon: {epsilon} < epsilon_hat: {ilp.prob.value}\n"
    err_str += f"Opt Var:\n{ilp.opt_var.value}\n"
    return err_str


if __name__ == '__main__':
    unittest.main()
