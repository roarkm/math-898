import unittest
from src.algorithms.certify import Certify
from src.algorithms.test_helper import identity_test_eps_rob


class TestCertify(unittest.TestCase):

    def test_eps_robustness(self):
        identity_test_eps_rob(self, Certify, nn_depth=2,
                              x=[[9], [1]], eps=1,
                              expect_robustness=True)
        identity_test_eps_rob(self, Certify, nn_depth=2,
                              x=[[4], [4.00001]], eps=1,
                              expect_robustness=False)


if __name__ == '__main__':
    unittest.main()
