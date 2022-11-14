import unittest
from src.algorithms.certify import Certify
from src.algorithms.test_helper import identity_test


class TestCertify(unittest.TestCase):

    def test_identity(self):
        identity_test(self, Certify)


if __name__ == '__main__':
    unittest.main()
