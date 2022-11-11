import unittest
import torch
import numpy as np
import sys
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.test_helper import (identity_test_eps_rob,
                                        identity_test)

class TestIteratedLinearVerifier(unittest.TestCase):

    def test_identity(self):
        identity_test(self, IteratedLinearVerifier)

if __name__ == '__main__':
    unittest.main()
