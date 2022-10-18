import unittest
import torch
import numpy as np
import sys
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.test_helper import (identity_test,
                                        debug_failure)

class TestIteratedLinearVerifier(unittest.TestCase):

    def test_id(self):
        identity_test(self, IteratedLinearVerifier, dim=2, nn_depth=2, eps=1)
        identity_test(self, IteratedLinearVerifier, dim=5, nn_depth=2, eps=1)


if __name__ == '__main__':
    unittest.main()
