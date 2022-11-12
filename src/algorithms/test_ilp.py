import unittest
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.test_helper import identity_test


class TestIteratedLinearVerifier(unittest.TestCase):

    def test_identity(self):
        identity_test(self, IteratedLinearVerifier)


if __name__ == '__main__':
    unittest.main()
