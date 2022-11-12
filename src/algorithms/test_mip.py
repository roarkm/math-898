import unittest
from src.algorithms.mip import MIPVerifier
from src.algorithms.test_helper import (identity_test_eps_rob,
                                        identity_test)

class TestMIPVerifier(unittest.TestCase):

    def test_identity(self):
        identity_test(self, MIPVerifier)

if __name__ == '__main__':
    unittest.main()

