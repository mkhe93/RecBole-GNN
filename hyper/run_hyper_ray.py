# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29, 2022/7/13, 2022/7/18
# @Author : Zihan Lin, Yupeng Hou, Gaowei Zhang, Lei Wang
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, zgw15630559577@163.com, zxcptss@gmail.com

import argparse
import os
import numpy as np

from recbole_gnn.quick_start import objective_function
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import math

def short_dirname(trial):
    return "trial_" + str(trial.trial_id)

def ray_tune(args):
    local_dir = r"/ray_log"
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    config_file_list = (
        [os.path.join(os.getcwd(), file) for file in config_file_list]
        if args.config_files
        else None
    )
    params_file = (
        os.path.join(os.getcwd(), args.params_file) if args.params_file else None
    )
    ray.init(num_cpus=24, runtime_env={"working_dir": local_dir})
    tune.register_trainable("train_func", objective_function)
    config = {}
    with open(params_file, "r") as fp:
        for line in fp:
            para_list = line.strip().split(" ")
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = (
                para_list[0],
                para_list[1],
                "".join(para_list[2:]),
            )
            if para_type == "choice":
                para_value = eval(para_value)
                config[para_name] = tune.choice(para_value)
            elif para_type == "uniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.uniform(float(low), float(high))
            elif para_type == "quniform":
                low, high, q = para_value.strip().split(",")
                config[para_name] = tune.quniform(float(low), float(high), float(q))
            elif para_type == "loguniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.loguniform(
                    math.exp(float(low)), math.exp(float(high))
                )
            else:
                raise ValueError("Illegal param type [{}]".format(para_type))

    # choose different schedulers to use different tuning optimization algorithms
    # For details, please refer to Ray's official website https://docs.ray.io
    scheduler = ASHAScheduler(
        metric="best_valid_score", mode="max", max_t=100, grace_period=1, reduction_factor=2
    )

    log_dir = r"/ray_log"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config,
        num_samples=5,
        log_to_file=args.output_file,
        scheduler=scheduler,
        storage_path=log_dir,
        resources_per_trial={"cpu": 2},
        trial_dirname_creator=short_dirname
    )

    best_trial = result.get_best_trial("ndcg@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_files", type=str, default="hyper.yaml", help="fixed config files"
    )
    parser.add_argument("--params_file", type=str, default="params.hyper", help="parameters file")
    parser.add_argument(
        "--output_file", type=str, default="forwardGNN-hyper-ray.result", help="output file"
    )
    parser.add_argument(
        "--display_file", type=str, default=None, help="visualization file"
    )
    parser.add_argument("--tool", type=str, default="Ray", help="tuning tool")
    args, _ = parser.parse_known_args()

    if args.tool == "Ray":
        ray_tune(args)
    else:
        raise ValueError(f"The tool [{args.tool}] should in ['Hyperopt', 'Ray']")
