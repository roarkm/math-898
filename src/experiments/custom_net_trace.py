from src.algorithms.certify import Certify
from src.algorithms.ilp import (LinearVerifier,
                                IteratedLinearVerifier)
from src.algorithms.mip import MIPVerifier
import torch
import numpy as np
from src.models.multi_layer import (custom_net,
                                    MultiLayerNN)


def trace_ilp_iter(f, x, eps):
    ilp = IteratedLinearVerifier(f)
    e_robust = ilp.decide_eps_robustness(x, eps)
    print(f"\nILP: {f.name} is ({eps})-robust at {x}?  {e_robust}")
    if not e_robust:
        ce = ilp.counter_example
        ce_class = f.class_for_input(ce)
        print(f"Counter-example found: f({ce}) = \
                {f(torch.tensor(ce).T.float()).detach().numpy()} \
                |--> {ce_class}")
    # print(ilp.str_constraints())


def trace_ilp(f, x, eps):
    ilp = LinearVerifier(f)
    e_robust = ilp.decide_eps_robustness(x, eps)
    print(f"\nILP: {f.name} is ({eps})-robust at {x}?  {e_robust}")
    if not e_robust:
        ce = ilp.counter_example
        ce_class = f.class_for_input(ce)
        print(f"Counter-example found: f({ce}) = \
                {f(torch.tensor(ce).T.float()).detach().numpy()} \
                |--> {ce_class}")
    print(ilp.str_opt_soln())
    print(ilp.str_constraints())


def trace_mip(f, x, eps):
    mip = MIPVerifier(f)
    e_robust = mip.decide_eps_robustness(x, eps)
    print(f"\nNSVerify: {f.name} is ({eps})-robust at {x}?  {e_robust}")
    if not e_robust:
        ce = mip.counter_example
        ce_class = f.class_for_input(ce)
        print(f"Counter-example found: f({ce}) = \
                {f(torch.tensor(ce).T.float()).detach().numpy()} \
                |--> {ce_class}")
    print(mip.str_constraints())
    # print(mip.str_opt_soln())


def trace_sdp(f, x, eps):
    cert = Certify(f)
    e_robust = cert.decide_eps_robustness(x, eps)
    print(f"\nCertify: {f.name} is ({eps})-robust at {x}?  {e_robust}")
    # print(cert.P)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(cert.Q)
    exit()
    del cert
    cert = Certify(f)
    cert.build_symbolic_matrices(x, eps)


if __name__ == '__main__':
    f = custom_net()
    eps = 0.5
    x = [[1], [1]]

    x_class = f.class_for_input(x)
    print(f"f({x}) = {f(torch.tensor(x).T.float()).detach().numpy()}\
           |--> class {x_class}")

    # trace_ilp_iter(f, x, eps)
    trace_ilp(f, x, eps)
    # # trace_mip(f, x, eps)
    # trace_sdp(f, x, eps)
