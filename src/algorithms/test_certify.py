import unittest
from src.algorithms.certify import Certify
from src.algorithms.test_helper import (identity_test_eps_rob,
                                        ER_TEST_CASES)


class TestCertify(unittest.TestCase):

    def test_eps_robustness(self):
        for t in ER_TEST_CASES:
            identity_test_eps_rob(self, Certify, nn_depth=2,
                                  x=t['x'], eps=t['eps'],
                                  expect_robustness=t['expect'])


if __name__ == '__main__':
    unittest.main()
