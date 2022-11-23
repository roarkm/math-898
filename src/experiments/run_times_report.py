import pandas as pd

def read_log(log_file_name, meta_file_name):
    info = pd.read_csv(meta_file_name)
    df = pd.read_csv(log_file_name)
    # print(info.head())

    report_desc = f"{info['alg_name'][0]}, {info['exp_type'][0]} \n"
    report_desc += f"Network Depth: {info['depth'][0]}, Network Arch: {info['layer_widths'][0]} \n"
    report_desc += f"Number of Runs: {df.shape[0]}"
    print(report_desc)
    print(f"Avg Solve Time (sec): {df['solve_time'].mean().round(5)}")
    print(f"Avg Build Time (sec): {df['build_time'].mean().round(5)}")

if __name__ == '__main__':
    read_log('experiment_logs/ILP_ER_ea6bcfeb8b750bf20900d71a5b9aa942af283366_log.csv',
             'experiment_logs/ILP_ER_ea6bcfeb8b750bf20900d71a5b9aa942af283366_meta.csv')
