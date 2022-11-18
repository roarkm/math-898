import unittest
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.test_helper import (identity_test_eps_rob,
                                        identity_test_pw_rob,
                                        PW_TEST_CASES,
                                        ER_TEST_CASES)


class TestIteratedLinearVerifier(unittest.TestCase):

    def test_eps_robustness(self):
        for t in ER_TEST_CASES:
            identity_test_eps_rob(self, IteratedLinearVerifier, nn_depth=2,
                                  x=t['x'], eps=t['eps'],
                                  expect_robustness=t['expect'])

    def test_pw_robustness(self):
        for t in PW_TEST_CASES:
            identity_test_pw_rob(self, IteratedLinearVerifier, nn_depth=2,
                                 x=t['x'], expected_eps=t['expected_eps'])


if __name__ == '__main__':
    unittest.main()
