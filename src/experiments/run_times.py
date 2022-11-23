from src.models.multi_layer import MultiLayerNN, identity_map
from src.algorithms.ilp import IteratedLinearVerifier as ILP
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cvxpy as cp
import torch
import numpy as np
import math
import csv
import hashlib
import time


def experiment_hash(layer_widths, max_iters, solver):
    data = layer_widths + [max_iters] + [solver]
    return hashlib.sha1(str(data).encode('utf-8')).hexdigest()


def log_file_name(algo_name, alg_type, layer_widths, max_iters, solver):
    return f"{algo_name}_{alg_type}_{experiment_hash(layer_widths, max_iters, solver)}_log.csv"


def meta_file_name(algo_name, alg_type, layer_widths, max_iters, solver):
    return f"{algo_name}_{alg_type}_{experiment_hash(layer_widths, max_iters, solver)}_meta.csv"


def random_x():
    return [[1], [1.001]]


def run_exp(VAlg, num_runs, alg_type):
    f = identity_map(2, 2)
    alg = VAlg(f)

    meta_fieldnames = {
        'alg_name': alg.name,
        'max_iters': alg.max_iters,
        'solver': alg.solver,
        'exp_type': alg_type,
        'layer_widths': '_'.join([str(d) for d in f.weight_dims()]),
        'depth': len(f.weight_dims())
    }
    meta_file = meta_file_name(alg.name, alg_type,
                               f.weight_dims(), alg.max_iters, alg.solver)
    exp_dir = "experiment_logs"
    with open(f"{exp_dir}/{meta_file}",
              'w', encoding='UTF8', newline='') as meta_f:
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

    with open(f"{exp_dir}/{log_file}",
              'w', encoding='UTF8', newline='') as log_f:
        writer = csv.DictWriter(log_f, fieldnames=exp_features)
        writer.writeheader()
        for i in range(num_runs):
            alg = VAlg(f)
            x = random_x()
            eps = 0.1

            build_start = time.time()
            alg._build_eps_robustness_problem(x, 0.1)
            build_end = time.time()
            bt = build_end - build_start

            solve_start = time.time()
            alg._decide_eps_robustness()
            solve_end = time.time()
            st = solve_end - solve_start

            result = {
                'x': '_'.join([str(d[0]) for d in x]),
                'eps': eps,
                'build_time': bt,
                'solve_time': st,
                'warnings': 'NA',
                'errors': 'NA',
                'result': True,
            }
            writer.writerow(result)

            del alg
            del x


if __name__ == '__main__':
    run_exp(ILP, num_runs=100, alg_type='ER')
