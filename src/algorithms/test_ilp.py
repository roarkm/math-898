import unittest
import torch
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.abstract_verifier import AbstractVerifier
from src.models.multi_layer import MultiLayerNN

class TestIteratedLinearVerifier(unittest.TestCase):

    # def test_constraints_for_seperating_hyperplane(self):
        # weights = [
            # [[1, 0],
             # [0, 1]],
            # [[1, 0],
             # [0, 1]],
        # ]
        # bias_vecs =[
            # (0,0),
            # (0,0),
        # ]
        # f = MultiLayerNN(weights, bias_vecs)
        # x = torch.rand((1,len(weights[0][0])))
        # assert torch.equal(x, f(x))


    def test_identity_network(self):
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
            self.assertTrue(ilp.verify_at_point(x=x, eps=epsilon),
                            f"Identity map should be {epsilon}-robust at {x}.")
            del ilp

        # TODO: Bug! if we call this code block before the above, said test fails!
        x = [[1], [1], [1]]
        for n in range(5):
            ilp = IteratedLinearVerifier(f)
            epsilon = 2**(-n)
            r = ilp.verify_at_point(x=x, eps=epsilon)
            if r:
                err = f"Identity map should not be robust at {x}.\n"
                err += f"Epsilon: {epsilon} < epsilon_hat: {ilp.prob.value}\n"
                err += f"Opt Var: {ilp.opt_var.value}\n"
                self.assertFalse(True, err)


    # def test_two_layer_identity_network(self):
        # weights = [
            # [[1, 0],
             # [0, 1]],
            # [[1, 0],
             # [0, 1]],
        # ]
        # bias_vecs =[
            # (0,0),
            # (0,0),
        # ]
        # f = MultiLayerNN(weights, bias_vecs)
        # x = torch.rand((1,2))
        # assert torch.equal(x, f(x))

        # # this fails if called after the decision boundary test!
        # x = [[9], [0]]
        # for n in range(1, 3):
            # # SCS hangs for eps >= 4
            # ilp = IteratedLinearVerifier(f)
            # epsilon = n + 0.3
            # self.assertTrue(ilp.verify_at_point(x=x, eps=epsilon),
                            # f"Identity f should be {epsilon}-robust at {x}.")
            # del ilp

        # # TODO: Bug! if we call this code block before the above, said test fails!
        # x = [[1], [1]]
        # for n in range(5):
            # ilp = IteratedLinearVerifier(f)
            # epsilon = 2**(-n)
            # self.assertFalse(ilp.verify_at_point(x=x, eps=epsilon),
                             # f"Identity f should never be {epsilon}-robust at {x} (on the decision boundary).")


if __name__ == '__main__':
    unittest.main()
