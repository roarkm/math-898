import torch
import numpy as np
from src.models.multi_layer import (MultiLayerNN,
                                    null_map,
                                    identity_map)


def identity_test(tc, VerifAlg, dim, nn_depth, eps):
    # TODO: test combinations of parameterized dimension, depth
    f = identity_map(dim, nn_depth)
    verif_alg = VerifAlg(f)
    tol = 10**(-17)
    x = np.ones((dim, 1))

    r = verif_alg.verify_at_point(x=x, eps=eps)
    tc.assertFalse(r, debug_failure("\nIdentity map SHOULD NOT be robust",
                                    x, eps, verif_alg))

    x[0,0] = 9
    r = verif_alg.verify_at_point(x=x, eps=eps)
    tc.assertTrue(r, debug_failure("\nIdentity map SHOULD be robust",
                                   x, eps, verif_alg))



def debug_failure(err_str, x, epsilon, verif_alg):
    im_adv = verif_alg.f(torch.tensor(verif_alg.free_vars('z0').value).T.float()).detach().numpy()
    fx     = verif_alg.f(torch.tensor(x).T.float()).detach().numpy()
    x_class   = np.argsort(fx)[0][-1]
    adv_class = np.argsort(im_adv)[0][-1]

    err_str += f" at {x}.\n"
    err_str += f"f({x}) = {fx} |--> {x_class}.\n"
    err_str += f"\nf({verif_alg.free_vars('z0').value}) = {im_adv} |--> {adv_class}\n"
    err_str += f"Epsilon: {epsilon} < epsilon_hat: {verif_alg.prob.value}\n"
    # err_str += f"Opt Var:\n{verif_alg.free_vars('z0').value}\n"
    return err_str
