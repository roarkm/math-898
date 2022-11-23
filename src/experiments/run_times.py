from src.models.multi_layer import MultiLayerNN, identity_map
from src.algorithms.ilp import IteratedLinearVerifier as ILP
from src.algorithms.certify import Certify
from src.algorithms.mip import MIPVerifier
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import cvxpy as cp
# import torch
# import numpy as np
# import math
import csv
import hashlib
import time


def experiment_hash(layer_widths, max_iters, solver):
    data = layer_widths + [max_iters] + [solver]
    return hashlib.sha1(str(data).encode('utf-8')).hexdigest()


def _experiment_string(algo_name, alg_type, layer_widths, max_iters, solver):
    return f"{algo_name}_{alg_type}_{solver}_{len(layer_widths)}"


def log_file_name(algo_name, alg_type, layer_widths, max_iters, solver):
    s = _experiment_string(algo_name, alg_type, layer_widths, max_iters, solver)
    return f"{s}_log.csv"


def meta_file_name(algo_name, alg_type, layer_widths, max_iters, solver):
    s = _experiment_string(algo_name, alg_type, layer_widths, max_iters, solver)
    return f"{s}_meta.csv"


def random_x(width):
    x = []
    for _ in range(width-1):
        x.append([2])
    x.append([9])
    return x


def run_exp(VAlg, num_runs, alg_type):
    x_dim = 2
    f = identity_map(x_dim, 3)
    alg = VAlg(f)

    meta_fieldnames = {
        'alg_name': alg.name,
        'exp_type': alg_type,
        'solver': alg.solver,
        'depth': len(f.weight_dims()),
        'layer_widths': '-'.join([str(d) for d in f.weight_dims()]),
        'max_iters': alg.max_iters,
    }
    meta_file = meta_file_name(alg.name, alg_type,
                               f.weight_dims(), alg.max_iters, alg.solver)
    exp_dir = "experiment_logs"
    with open(f"{exp_dir}/{meta_file}", 'w',
              encoding='UTF8', newline='') as meta_f:
        writer = csv.DictWriter(meta_f, fieldnames=meta_fieldnames.keys())
        writer.writeheader()
        writer.writerow(meta_fieldnames)

    log_file = log_file_name(alg.name, alg_type,
                             f.weight_dims(), alg.max_iters, alg.solver)
    exp_features = [
        'x',
        'eps',
        'build_time',
        'solve_time',
        'warnings',
        'errors',
        'result',
    ]
    del alg

    with open(f"{exp_dir}/{log_file}", 'w',
              encoding='UTF8', newline='') as log_f:
        writer = csv.DictWriter(log_f, fieldnames=exp_features)
        writer.writeheader()
        for i in range(num_runs):
            alg = VAlg(f)
            x = random_x(x_dim)
            eps = 0.1

            build_start = time.time()
            alg._build_eps_robustness_problem(x, 0.1)
            build_end = time.time()
            bt = build_end - build_start

            solve_start = time.time()
            res = alg._decide_eps_robustness()
            solve_end = time.time()
            st = solve_end - solve_start

            result = {
                'x': '_'.join([str(d[0]) for d in x]),
                'eps': eps,
                'build_time': bt,
                'solve_time': st,
                'warnings': 'NA',
                'errors': 'NA',
                'result': res,
            }
            print(f"{alg.name} - solved {i}")
            print(result)
            writer.writerow(result)

            del alg
            del x


if __name__ == '__main__':
    n=100
    run_exp(ILP, num_runs=n, alg_type='ER')
    run_exp(MIPVerifier, num_runs=n, alg_type='ER')
    run_exp(Certify, num_runs=n, alg_type='ER')
