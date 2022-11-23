import pandas as pd
import os
import re

def report_log(log_file_name, meta_file_name):
    info = pd.read_csv(meta_file_name)
    df = pd.read_csv(log_file_name)
    # print(info.head())

    report_desc = f"{info['alg_name'][0]}, {info['exp_type'][0]} \n"
    report_desc += f"Network Depth: {info['depth'][0]}, Network Arch: {info['layer_widths'][0]} \n"
    report_desc += f"Number of Runs: {df.shape[0]}\n"
    report_desc += f"Avg Build Time (sec): {df['build_time'].mean().round(5)}\n"
    report_desc += f"Avg Solve Time (sec): {df['solve_time'].mean().round(5)}\n"
    return report_desc


if __name__ == '__main__':
    d = os.path.join(os.getcwd(), 'experiment_logs')
    for filename in os.listdir(d):
        f = os.path.join(d, filename)
        if os.path.isfile(f):
            fn, ext = os.path.splitext(f)
            if fn.endswith('_log'):
                exp_name = '_'.join(fn.split('_')[:-1])
                print(report_log(f"{exp_name}_log.csv", f"{exp_name}_meta.csv"))
