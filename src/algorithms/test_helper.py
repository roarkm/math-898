import torch
import numpy as np
from src.models.multi_layer import (MultiLayerNN,
                                    null_map,
                                    identity_map)


def identity_test_eps_rob(tc, VerifAlg, x, eps, nn_depth, expect):
    dim = len(x)
    f = identity_map(dim, nn_depth)
    verif_alg = VerifAlg(f)

    r = verif_alg.decide_eps_robustness(x=x, eps=eps)
    tc.assertTrue(r == expect,
                  debug_eps_rob_failure(expect, x, eps, verif_alg))


def debug_eps_rob_failure(expect, x, eps, v_alg):
    fx = v_alg.f(torch.tensor(x).float().T).detach().numpy()
    x_class   = np.argsort(fx)[0][-1]

    desc = "Should"
    if not expect:
        desc += " Not"

    err_str = f"\n{v_alg.f.name} {desc} be ({eps})-robust at: \n"
    err_str += f"{x}\n |--> {x_class}\n"
    return err_str
