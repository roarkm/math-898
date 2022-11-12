import unittest
from src.algorithms.certify import Certify
from src.algorithms.test_helper import identity_test


class TestCertify(unittest.TestCase):

    def test_identity(self):
        identity_test(self, Certify)


if __name__ == '__main__':
    unittest.main()

# import unittest
# import torch
# from src.models.multi_layer import MultiLayerNN, null_map, identity_map
# from src.algorithms.certify import Certify


# class TestCertify(unittest.TestCase):

    # def test_null_map(self):
        # f = null_map(2, 2)
        # x = torch.rand((1,2))
        # y = torch.zeros((1,2))
        # assert torch.equal(y, f(x))

        # # this fails if called after the decision boundary test!
        # x = [[9], [0]]
        # for n in range(1, 3):
            # cert = Certify(f)
            # epsilon = n + 0.3
            # self.assertTrue(cert.verify_at_point(x=x, eps=epsilon),
                            # f"Null map should always be {epsilon}-robust at {x}.")
            # del cert


    # def test_two_layer_identity_network(self):
        # dim = 2
        # f = identity_map(dim, 2)
        # x = torch.rand((1,dim))
        # assert torch.equal(x, f(x))

        # # this fails if called after the decision boundary test!
        # x = [[9], [0]]
        # for n in range(1, 3):
            # # SCS hangs for eps >= 4
            # cert = Certify(f)
            # epsilon=n + 0.3
            # self.assertTrue(cert.verify_at_point(x=x, eps=epsilon),
                            # f"Identity f should be {epsilon}-robust at {x}.")
            # del cert

        # # TODO: Bug! if we call this code block before the above, said test fails!
        # x = [[1], [1]]
        # for n in range(5):
            # cert = Certify(f)
            # epsilon = 2**(-n)
            # self.assertFalse(cert.verify_at_point(x=x, eps=epsilon),
                             # f"Identity f should never be {epsilon}-robust at {x} (on the decision boundary).")


# if __name__ == '__main__':
    # unittest.main()
