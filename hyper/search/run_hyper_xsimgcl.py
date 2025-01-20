import argparse
import torch

from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='../parameter_fixed/hyper_xsimgcl_djc100000.yaml', help='fixed config files')
    parser.add_argument('--params_file', type=str, default='../parameter_space/params_xsimgcl.hyper', help='parameters file')
    parser.add_argument('--output_file', type=str, default='../results/xsimgcl-hyper-djc100000-valid-splitk.result', help='output file')
    args, _ = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set. Others: 'bayes','random'
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hp = HyperTuning(objective_function, algo='bayes', max_evals=200, early_stop=50,
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    torch.set_num_threads(8)
    hp.run()
    hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

if __name__ == '__main__':
    main()


# Try on different embedding sizes : embedding_size choice [32, 64, 92, 128, 256]
# C:\Users\s8347434\Documents\RecBole-GNN\.venv\Scripts\python.exe C:\Users\s8347434\Documents\RecBole-GNN\hyper\search\run_hyper_xsimgcl.py
# running parameters:
# {'embedding_size': 64, 'epochs': 100, 'eps': 0.2, 'lambda': 1e-05, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.1}
# current best valid score: 0.1227
# current best valid result:
# OrderedDict({'precision@10': 0.0958, 'hit@10': 0.4876, 'mrr@10': 0.2395, 'ndcg@10': 0.1227, 'map@10': 0.064, 'itemcoverage@10': 0.0632, 'averagepopularity@10': 71.9181, 'tailpercentage@10': 0.001})
# current test result:
# OrderedDict({'precision@10': 0.0475, 'hit@10': 0.3232, 'mrr@10': 0.1385, 'ndcg@10': 0.0712, 'map@10': 0.0333, 'itemcoverage@10': 0.062, 'averagepopularity@10': 72.402, 'tailpercentage@10': 0.0005})
# running parameters:
# {'embedding_size': 128, 'epochs': 100, 'eps': 0.2, 'lambda': 0.01, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# current best valid score: 0.1639
# current best valid result:
# OrderedDict({'precision@10': 0.1275, 'hit@10': 0.5477, 'mrr@10': 0.2916, 'ndcg@10': 0.1639, 'map@10': 0.0951, 'itemcoverage@10': 0.0967, 'averagepopularity@10': 54.7497, 'tailpercentage@10': 0.0087})
# current test result:
# OrderedDict({'precision@10': 0.0599, 'hit@10': 0.3674, 'mrr@10': 0.1604, 'ndcg@10': 0.0911, 'map@10': 0.0456, 'itemcoverage@10': 0.094, 'averagepopularity@10': 55.3409, 'tailpercentage@10': 0.0076})
# running parameters:
# {'embedding_size': 64, 'epochs': 100, 'eps': 0.2, 'lambda': 0.005, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.1}
# running parameters:
# {'embedding_size': 128, 'epochs': 100, 'eps': 0.2, 'lambda': 0.05, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.2}
# current best valid score: 0.1924
# current best valid result:
# OrderedDict({'precision@10': 0.1484, 'hit@10': 0.5811, 'mrr@10': 0.3279, 'ndcg@10': 0.1924, 'map@10': 0.1192, 'itemcoverage@10': 0.1369, 'averagepopularity@10': 40.5353, 'tailpercentage@10': 0.0145})
# current test result:
# OrderedDict({'precision@10': 0.0707, 'hit@10': 0.4086, 'mrr@10': 0.177, 'ndcg@10': 0.1125, 'map@10': 0.0599, 'itemcoverage@10': 0.135, 'averagepopularity@10': 40.1098, 'tailpercentage@10': 0.0138})
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.2, 'lambda': 0.1, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 32, 'epochs': 100, 'eps': 0.1, 'lambda': 0.01, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.05}
# running parameters:
# {'embedding_size': 64, 'epochs': 100, 'eps': 0.1, 'lambda': 0.1, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 64, 'epochs': 100, 'eps': 0.2, 'lambda': 0.005, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.05}
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.1, 'lambda': 1e-07, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.2, 'lambda': 1e-06, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.1, 'lambda': 1e-06, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.2, 'lambda': 0.01, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.1, 'lambda': 0.05, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 128, 'epochs': 100, 'eps': 0.2, 'lambda': 0.1, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.1}
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.2, 'lambda': 0.0001, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.1}
# running parameters:
# {'embedding_size': 32, 'epochs': 100, 'eps': 0.2, 'lambda': 0.01, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 256, 'epochs': 100, 'eps': 0.1, 'lambda': 1e-05, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.1}
# running parameters:
# {'embedding_size': 32, 'epochs': 100, 'eps': 0.1, 'lambda': 0.005, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.1}
# running parameters:
# {'embedding_size': 64, 'epochs': 100, 'eps': 0.2, 'lambda': 0.01, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.1}
# running parameters:
# {'embedding_size': 64, 'epochs': 100, 'eps': 0.2, 'lambda': 0.05, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.1}
# running parameters:
# {'embedding_size': 128, 'epochs': 100, 'eps': 0.2, 'lambda': 0.05, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.05}
# running parameters:
# {'embedding_size': 92, 'epochs': 100, 'eps': 0.2, 'lambda': 0.1, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 128, 'epochs': 100, 'eps': 0.2, 'lambda': 1e-07, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 92, 'epochs': 100, 'eps': 0.2, 'lambda': 0.0001, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 2, 'reg_weight': 0.0001, 'temperature': 0.2}
# running parameters:
# {'embedding_size': 128, 'epochs': 100, 'eps': 0.2, 'lambda': 0.05, 'layer_cl': 1, 'learning_rate': 0.002, 'n_layers': 3, 'reg_weight': 0.0001, 'temperature': 0.05}
#  12%|█▏        | 24/200 [39:39:47<276:47:48, 5661.75s/trial, best loss: -0.1924]