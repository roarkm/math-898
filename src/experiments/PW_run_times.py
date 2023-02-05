from src.models.multi_layer import (MultiLayerNN,
                                    identity_map,
                                    custom_net)
from src.algorithms.ilp import LinearVerifier as LP
from src.algorithms.ilp import IteratedLinearVerifier as ILP
from src.algorithms.certify import Certify
from src.algorithms.mip import MIPVerifier
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import cvxpy as cp
# import torch
import numpy as np
# import math
import csv
import hashlib
import time
import os.path
import sys, getopt


def experiment_hash(layer_widths, max_iters, solver):
    data = layer_widths + [max_iters] + [solver]
    return hashlib.sha1(str(data).encode('utf-8')).hexdigest()


def _experiment_string(algo_name, alg_type, layer_widths, max_iters, solver):
    return f"{algo_name}_{alg_type}_{solver}_{len(layer_widths)}"


def log_file_name(algo_name, alg_type, layer_widths, max_iters, solver):
    s = _experiment_string(algo_name, alg_type,
                           layer_widths, max_iters, solver)
    return f"{s}_log.csv"


def meta_file_name(algo_name, alg_type, layer_widths, max_iters, solver):
    s = _experiment_string(algo_name, alg_type,
                           layer_widths, max_iters, solver)
    return f"{s}_meta.csv"


def random_x(width, _min, _max):
    x = []
    for _ in range(width):
        x.append([np.random.uniform(_min, _max)])
    return x


def run_exp(VAlg, alg_type, f, x):

    alg = VAlg(f)
    meta_fieldnames = {
        'alg_name': alg.name,
        'map_name': f.name,
        'exp_type': alg_type,
        'solver': alg.solver,
        'depth': len(f.weight_dims()),
        'layer_widths': '-'.join([str(d) for d in f.weight_dims()]),
        'max_iters': alg.max_iters,
    }
    meta_file = meta_file_name(alg.name, alg_type,
                               f.weight_dims(), alg.max_iters, alg.solver)
    exp_dir = "experiment_logs/PW"
    mf = f"{exp_dir}/{meta_file}"
    if not os.path.exists(mf):
        with open(mf, 'w', encoding='UTF8', newline='') as meta_f:
            writer = csv.DictWriter(meta_f, fieldnames=meta_fieldnames.keys())
            writer.writeheader()
            writer.writerow(meta_fieldnames)

    log_file = log_file_name(alg.name, alg_type,
                             f.weight_dims(), alg.max_iters, alg.solver)
    exp_features = [
        'x',
        'build_time',
        'solve_time',
        'adv_example',
    ]
    del alg

    lf = f"{exp_dir}/{log_file}"
    found_file = os.path.exists(lf)
    with open(lf, 'a', encoding='UTF8', newline='') as log_f:
        writer = csv.DictWriter(log_f, fieldnames=exp_features)
        if not found_file:
            writer.writeheader()

        alg = VAlg(f)

        build_start = time.time()
        alg._build_pw_robustness(x)
        build_end = time.time()
        bt = build_end - build_start

        solve_start = time.time()
        adv_ex = alg._compute_robustness()
        solve_end = time.time()
        st = solve_end - solve_start

        if alg.counter_example is not None:
            adv_ex = '_'.join([str(d[0]) for d in adv_ex.numpy().list()]),
            print(ce)
        else:
            ce = None

        result = {
            'x': '_'.join([str(d[0]) for d in x]),
            'build_time': bt,
            'solve_time': st,
            'adv_example': ce,
        }
        print(f"{alg.name}")
        print(result)
        writer.writerow(result)

        del alg
        del x


def main(x_dim):
    print(f"width/depth: {x_dim}")

    _min = 1
    _max = 3
    f = identity_map(x_dim, x_dim)
    x = random_x(x_dim, _min, _max)

    run_exp(LP, alg_type='PW', f=f, x=x)
    run_exp(MIPVerifier, alg_type='PW', f=f, x=x)


if __name__ == '__main__':
    x_dim = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:", ["dim="])
    except getopt.GetoptError:
        print('run_times.py -e 0|1')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-d' or opt == '--dim':
            x_dim = int(arg)

    if x_dim is None:
        raise Exception("Must pass argument '-d {int}'")
    main(x_dim)
