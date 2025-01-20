# @Time   : 2023/2/13
# @Author : Gaowei Zhang
# @Email  : zgw2022101006@ruc.edu.cn


import argparse
import pandas as pd
from pathlib import Path

from tqdm import tqdm
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset
from recbole_gnn.data.dataset_metrics import GraphDatasetEvaluator

if __name__ == "__main__":

    dataset_eval_list = []

    for i in tqdm(range(114,115)):
        file_path = Path(f"../asset/data/real-life-atomic-splits/real-life-atomic-100000-{i}/real-life-atomic-100000-{i}.inter")
        if not file_path.exists():
            break

        parser = argparse.ArgumentParser()
        parser.add_argument("--config_files", type=str, default="config_files/datasets.yaml", help="config files")
        args, _ = parser.parse_known_args()

        # configurations initialization
        config_file_list = args.config_files.strip().split(",")
        config = Config(model="BPR", dataset=f"real-life-atomic-100000-{i}", config_file_list=config_file_list)

        # dataset filtering
        dataset = create_dataset(config)

        # calculate dataset metrics
        dataset_evaluator = GraphDatasetEvaluator(config, dataset)
        dataset_eval_dict = {"dataset": f"real-life-atomic-100000-{i}"}
        dataset_eval_dict.update(dataset_evaluator.evaluate())

        dataset_eval_list.append(dataset_eval_dict)

    df = pd.DataFrame(dataset_eval_list)

    #df.to_csv('log/Dataset/dataset_eval.csv', sep='\t', index=False)

