import unittest
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.test_helper import (identity_test_eps_rob,
                                        identity_test_pw_rob)


class TestIteratedLinearVerifier(unittest.TestCase):

    def test_eps_robustness(self):
        identity_test_eps_rob(self, IteratedLinearVerifier, nn_depth=2,
                              x=[[9], [1]], eps=1,
                              expect_robustness=True)
        identity_test_eps_rob(self, IteratedLinearVerifier, nn_depth=2,
                              x=[[4], [4.00001]], eps=1,
                              expect_robustness=False)

    def test_pw_robustness(self):
        identity_test_pw_rob(self, IteratedLinearVerifier, nn_depth=2,
                             x=[[4], [4.0001]], expected_eps=0.000005)
        identity_test_pw_rob(self, IteratedLinearVerifier, nn_depth=2,
                             x=[[9], [1]], expected_eps=4)


if __name__ == '__main__':
    unittest.main()
