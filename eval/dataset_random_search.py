# @Time   : 2023/2/13
# @Author : Gaowei Zhang
# @Email  : zgw2022101006@ruc.edu.cn
import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import argparse
import pandas as pd
from pathlib import Path
import threadpoolctl
import numpy as np
import torch
from tqdm import tqdm
from recbole_gnn.quick_start import run_recbole_gnn

if __name__ == "__main__":

    #np.random.seed(100)
    #random_numbers = np.random.randint(1, 177, size=10)

    random_numbers = [9, 25, 68, 104, 88, 80, 139, 95, 99, 54]

    model_list = [('UltraGCN', 'ultragcn')]

    for model in model_list:

        test_res_list = []
        test_res_best_user_list = []
        test_res_worst_user_list = []

        for i in tqdm(random_numbers, desc="Datasets", unit='datasets'):
            file_path = Path(f"../asset/data/real-life-atomic-splits/real-life-atomic-100000-{i}/real-life-atomic-100000-{i}.inter")
            if not file_path.exists():
                break

            if model[0] != 'ALS':
                torch.set_num_threads(8)

            parser = argparse.ArgumentParser()
            dataset = f"real-life-atomic-100000-{i}"
            parser.add_argument("--config_files", type=str, default=f"config_files/{model[1]}.yaml", help="config files")
            parser.add_argument('--dataset', '-d', type=str, default=dataset, help='name of datasets')
            parser.add_argument('--model', '-m', type=str, default=model[0], help='name of models')
            args, _ = parser.parse_known_args()

            # configurations initialization
            config_file_list = args.config_files.strip().split(' ') if args.config_files else None
            with threadpoolctl.threadpool_limits(1,"blas"):  # Due to a warning that occurred while running the ALS algorithm, important line!
                result = run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

            # calculate dataset metrics
            test_res_dict = {"Model": model[0], "dataset": f"real-life-atomic-100000-{i}"}
            test_res_best_user_dict = {"Model": model[0], "dataset": f"real-life-atomic-100000-{i}"}
            test_res_worst_user_dict = {"Model": model[0], "dataset": f"real-life-atomic-100000-{i}"}

            test_res_dict.update(result["test_result"])
            test_res_best_user_dict.update(result["best_user_evaluation"])
            test_res_worst_user_dict.update(result["worst_user_evaluation"])

            test_res_list.append(test_res_dict)
            test_res_best_user_list.append(test_res_best_user_dict)
            test_res_worst_user_list.append(test_res_worst_user_dict)

            # Combine all dictionaries into a single list
            combined_results = []

            for main_dict, best_dict, worst_dict in zip(test_res_list, test_res_best_user_list, test_res_worst_user_list):
                # Combine dictionaries with the desired prefixes
                combined_entry = {}
                combined_entry.update(main_dict)  # No prefix for the main dictionary
                combined_entry.update(
                    {f"best_user_{key}": value for key, value in best_dict.items() if key not in main_dict})
                combined_entry.update(
                    {f"worst_user_{key}": value for key, value in worst_dict.items() if key not in main_dict})

                # Append the combined entry to the results
                combined_results.append(combined_entry)

            # Convert the combined results into a DataFrame (optional)
            df = pd.DataFrame(combined_results)

            if model[0] == "AsymKNN":
                df.to_csv(f'log/Benchmark/RO/{model[1]}-Benchmark-RO.csv', sep='\t', index=False)
            else:
                df.to_csv(f'log/Benchmark/RO/{model[0]}-10RandomSearch-RO.csv', sep='\t', index=False)
