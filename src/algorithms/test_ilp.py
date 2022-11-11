import unittest
import torch
import numpy as np
import sys
from src.algorithms.ilp import IteratedLinearVerifier
from src.algorithms.test_helper import identity_test_eps_rob

class TestIteratedLinearVerifier(unittest.TestCase):

    def test_id(self):
        identity_test_eps_rob(self, IteratedLinearVerifier, nn_depth=2,
                             x=[[9], [4]], eps=1, expect=True)


if __name__ == '__main__':
    unittest.main()
