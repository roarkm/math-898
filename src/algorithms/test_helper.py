import torch
import numpy as np
from src.models.multi_layer import identity_map

ER_TEST_CASES = [
    {
         'x': [[9], [1]],
         'eps': 1,
         'expect': True
    },
    {
         'x': [[4], [4.5]],
         'eps':0.00000000001,
         'expect': True
    },
    {
         'x': [[4], [4.00001]],
         'eps': 1,
         'expect': False
    },
    # {  # if called again, this fails with Certify!
         # 'x': [[4], [4.5]],
         # 'eps':0.00000000001,
         # 'expect': True
    # },
]

PW_TEST_CASES = [
    {
        'x': [[9], [1]],
        'expected_eps': 4
    },
    {
        'x': [[4], [4.0001]],
        'expected_eps': 0.000005
    }
]


def identity_test_pw_rob(tc, VerifAlg, x, nn_depth, expected_eps):
    dim = len(x)
    f = identity_map(dim, nn_depth)
    verif_alg = VerifAlg(f)

    eps_hat = verif_alg.compute_robustness(x=x)
    try:
        tc.assertAlmostEqual(eps_hat, expected_eps, places=4)
    except AssertionError:
        debug_pw_rob_failure(expected_eps, x, eps_hat, verif_alg)
        del f
        del verif_alg


def debug_pw_rob_failure(expected_eps, x, eps_hat, v_alg):
    fx = v_alg.f(torch.tensor(x).float().T).detach().numpy()
    err_str = f"\n{v_alg.name} Test Failure"
    err_str += (f"\nExpected {v_alg.f.name} to have "
                f"nearest counter example at {expected_eps} distance\n")
    err_str += f"\tFrom {x} |--> {fx}\n"
    err_str += f"\tFound Counter Example: {v_alg.counter_example.T}\n"
    err_str += f"\tAt distance {eps_hat}."
    print(err_str)


def debug_eps_rob_failure(expect_robustness, x, eps, v_alg):
    fx = v_alg.f(torch.tensor(x).float().T).detach().numpy()
    x_class = np.argsort(fx)[0][-1]

    desc = "Should"
    if not expect_robustness:
        desc += " NOT"

    err_str = f"\n{v_alg.name} Test Failure"
    err_str += f"\n{v_alg.f.name} {desc} be ({eps})-robust at: \n"
    err_str += f"\t{x} |--> {x_class}\n"
    if expect_robustness:
        if (v_alg.name == 'ILP' or v_alg.name == 'NSVerify'):
            err_str += "\tFound Counterexample:\n"
            ce = torch.tensor(v_alg.counter_example).T.float()
            err_str += (f"\tf({v_alg.counter_example.T}) = "
                        f"{v_alg.f.forward(ce).detach().numpy()} |--> "
                        f"{v_alg.f.class_for_input(v_alg.counter_example)}")
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
        del f
        del verif_alg
