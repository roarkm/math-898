import torch
import numpy as np
from src.models.multi_layer import identity_map


def debug_eps_rob_failure(expect_robustness, x, eps, v_alg):
    fx = v_alg.f(torch.tensor(x).float().T).detach().numpy()
    x_class = np.argsort(fx)[0][-1]

    desc = "Should"
    if not expect_robustness:
        desc += " NOT"

    err_str = "\nTest Failure"
    err_str += f"\n{v_alg.f.name} {desc} be ({eps})-robust at: \n"
    err_str += f"\t{x} |--> {x_class}\n"
    if expect_robustness:
        if (v_alg.name == 'ILP' or v_alg.name == 'NSVerify'):
            err_str += f"\tFound Counterexample: {v_alg.counter_example.T}"
    print(err_str)


def identity_test_eps_rob(tc, VerifAlg, x, eps, nn_depth, expect_robustness):
    dim = len(x)
    f = identity_map(dim, nn_depth)
    verif_alg = VerifAlg(f)

    r = verif_alg.decide_eps_robustness(x=x, eps=eps)
    try:
        tc.assertEqual(r, expect_robustness)
    except AssertionError:
        debug_eps_rob_failure(expect_robustness, x, eps, verif_alg)


def identity_test(self, VerifAlg):
    identity_test_eps_rob(self, VerifAlg, nn_depth=2,
                          x=[[4], [4]], eps=1, expect_robustness=False)
    identity_test_eps_rob(self, VerifAlg, nn_depth=2,
                          x=[[9], [4]], eps=1, expect_robustness=True)
