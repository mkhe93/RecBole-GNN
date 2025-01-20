import argparse
import torch

from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='../parameter_fixed/hyper_ngcf_djc100000.yaml', help='fixed config files')
    parser.add_argument('--params_file', type=str, default='../parameter_space/params_ngcf.hyper', help='parameters file')
    parser.add_argument('--output_file', type=str, default='../results/ngcf-hyper-djc100000-leave-k.result', help='output file')
    args, _ = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set. Others: 'bayes','random'
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hp = HyperTuning(objective_function, algo='bayes', max_evals=200, early_stop=50,
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    torch.set_num_threads(12)
    hp.run()
    hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

if __name__ == '__main__':
    main()

# Try on different embedding sizes
# C:\Users\s8347434\Documents\RecBole-GNN\.venv\Scripts\python.exe C:\Users\s8347434\Documents\RecBole-GNN\hyper\search\run_hyper_ngcf.py
# running parameters:
# {'delay': 1e-05, 'epochs': 100, 'hidden_size_list': '[32,32,32]', 'learning_rate': 0.001, 'message_dropout': 0.2, 'node_dropout': 0.1, 'reg_weight': 1e-05}
#   0%|          | 0/200 [00:00<?, ?trial/s, best loss=?]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# current best valid score: 0.0706
# current best valid result:
# OrderedDict({'precision@10': 0.0549, 'hit@10': 0.3206, 'mrr@10': 0.1485, 'ndcg@10': 0.0706, 'map@10': 0.0336, 'itemcoverage@10': 0.1528, 'averagepopularity@10': 39.7415, 'tailpercentage@10': 0.0026})
# current test result:
# OrderedDict({'precision@10': 0.0296, 'hit@10': 0.225, 'mrr@10': 0.0851, 'ndcg@10': 0.0449, 'map@10': 0.0207, 'itemcoverage@10': 0.1483, 'averagepopularity@10': 43.4007, 'tailpercentage@10': 0.0039})
# running parameters:
# {'delay': 1e-05, 'epochs': 100, 'hidden_size_list': '[64]', 'learning_rate': 0.001, 'message_dropout': 0.3, 'node_dropout': 0.0, 'reg_weight': 0.0001}
# current best valid score: 0.0844
# current best valid result:
# OrderedDict({'precision@10': 0.0661, 'hit@10': 0.3836, 'mrr@10': 0.1734, 'ndcg@10': 0.0844, 'map@10': 0.0397, 'itemcoverage@10': 0.0398, 'averagepopularity@10': 90.1129, 'tailpercentage@10': 0.0})
# current test result:
# OrderedDict({'precision@10': 0.0322, 'hit@10': 0.2407, 'mrr@10': 0.0986, 'ndcg@10': 0.0504, 'map@10': 0.0227, 'itemcoverage@10': 0.0387, 'averagepopularity@10': 90.8287, 'tailpercentage@10': 0.0})
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[64,64]', 'learning_rate': 0.0005, 'message_dropout': 0.0, 'node_dropout': 0.2, 'reg_weight': 1e-05}
#   1%|          | 2/200 [2:32:15<244:57:57, 4453.93s/trial, best loss: -0.0844]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# current best valid score: 0.1017
# current best valid result:
# OrderedDict({'precision@10': 0.0824, 'hit@10': 0.4094, 'mrr@10': 0.1893, 'ndcg@10': 0.1017, 'map@10': 0.0526, 'itemcoverage@10': 0.1483, 'averagepopularity@10': 30.7354, 'tailpercentage@10': 0.0026})
# current test result:
# OrderedDict({'precision@10': 0.0417, 'hit@10': 0.279, 'mrr@10': 0.1099, 'ndcg@10': 0.0631, 'map@10': 0.03, 'itemcoverage@10': 0.1436, 'averagepopularity@10': 34.1781, 'tailpercentage@10': 0.0028})
# running parameters:
# {'delay': 0.01, 'epochs': 100, 'hidden_size_list': '[64,64,64,64]', 'learning_rate': 0.001, 'message_dropout': 0.2, 'node_dropout': 0.2, 'reg_weight': 0.0001}
#   2%|▏         | 3/200 [4:00:00<263:59:49, 4824.31s/trial, best loss: -0.1017]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[128,128,128,128]', 'learning_rate': 0.005, 'message_dropout': 0.0, 'node_dropout': 0.1, 'reg_weight': 0.0001}
#   2%|▏         | 4/200 [6:34:20<358:00:37, 6575.70s/trial, best loss: -0.1017]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# current best valid score: 0.1437
# current best valid result:
# OrderedDict({'precision@10': 0.1127, 'hit@10': 0.5038, 'mrr@10': 0.2553, 'ndcg@10': 0.1437, 'map@10': 0.0807, 'itemcoverage@10': 0.1117, 'averagepopularity@10': 55.167, 'tailpercentage@10': 0.0017})
# current test result:
# OrderedDict({'precision@10': 0.0538, 'hit@10': 0.332, 'mrr@10': 0.1448, 'ndcg@10': 0.083, 'map@10': 0.0407, 'itemcoverage@10': 0.108, 'averagepopularity@10': 53.4272, 'tailpercentage@10': 0.0017})
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[128,128,128,128]', 'learning_rate': 0.005, 'message_dropout': 0.0, 'node_dropout': 0.1, 'reg_weight': 1e-05}
#   2%|▎         | 5/200 [10:28:34<502:14:26, 9272.14s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.01, 'epochs': 100, 'hidden_size_list': '[128,128,128,128]', 'learning_rate': 0.01, 'message_dropout': 0.3, 'node_dropout': 0.1, 'reg_weight': 0.0001}
#   3%|▎         | 6/200 [14:28:30<593:32:57, 11014.32s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.1, 'epochs': 100, 'hidden_size_list': '[96]', 'learning_rate': 0.001, 'message_dropout': 0.0, 'node_dropout': 0.1, 'reg_weight': 0.0001}
#   4%|▎         | 7/200 [17:24:17<582:17:48, 10861.49s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 1e-05, 'epochs': 100, 'hidden_size_list': '[128,128]', 'learning_rate': 0.0005, 'message_dropout': 0.0, 'node_dropout': 0.1, 'reg_weight': 1e-05}
#   4%|▍         | 8/200 [17:59:47<431:02:10, 8081.93s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[64,64,64,64]', 'learning_rate': 0.0005, 'message_dropout': 0.2, 'node_dropout': 0.2, 'reg_weight': 0.0001}
#   4%|▍         | 9/200 [20:12:19<426:37:42, 8041.17s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[96]', 'learning_rate': 0.01, 'message_dropout': 0.0, 'node_dropout': 0.1, 'reg_weight': 1e-05}
#   5%|▌         | 10/200 [23:10:55<467:58:58, 8867.05s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[128,128]', 'learning_rate': 0.005, 'message_dropout': 0.3, 'node_dropout': 0.2, 'reg_weight': 0.0001}
#   6%|▌         | 11/200 [24:16:29<386:15:48, 7357.40s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 1e-05, 'epochs': 100, 'hidden_size_list': '[128,128]', 'learning_rate': 0.0001, 'message_dropout': 0.0, 'node_dropout': 0.2, 'reg_weight': 0.0001}
#   6%|▌         | 12/200 [26:33:17<397:44:02, 7616.18s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.01, 'epochs': 100, 'hidden_size_list': '[32,32]', 'learning_rate': 0.01, 'message_dropout': 0.1, 'node_dropout': 0.0, 'reg_weight': 0.0001}
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[96]', 'learning_rate': 0.001, 'message_dropout': 0.3, 'node_dropout': 0.1, 'reg_weight': 0.0001}
#   7%|▋         | 14/200 [29:28:42<324:08:57, 6273.86s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[96,96,96]', 'learning_rate': 0.01, 'message_dropout': 0.1, 'node_dropout': 0.1, 'reg_weight': 1e-05}
#   8%|▊         | 15/200 [30:31:01<283:07:39, 5509.51s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[32,32,32]', 'learning_rate': 0.0001, 'message_dropout': 0.3, 'node_dropout': 0.0, 'reg_weight': 1e-05}
# running parameters:
# {'delay': 0.1, 'epochs': 100, 'hidden_size_list': '[128,128,128]', 'learning_rate': 0.0005, 'message_dropout': 0.2, 'node_dropout': 0.1, 'reg_weight': 1e-05}
#   8%|▊         | 17/200 [34:22:26<314:47:50, 6192.74s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.1, 'epochs': 100, 'hidden_size_list': '[64,64,64]', 'learning_rate': 0.001, 'message_dropout': 0.3, 'node_dropout': 0.0, 'reg_weight': 1e-05}
# running parameters:
# {'delay': 0.1, 'epochs': 100, 'hidden_size_list': '[96]', 'learning_rate': 0.0005, 'message_dropout': 0.1, 'node_dropout': 0.2, 'reg_weight': 0.0001}
#  10%|▉         | 19/200 [39:18:32<373:34:14, 7430.14s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead
#
#
# running parameters:
# {'delay': 0.0001, 'epochs': 100, 'hidden_size_list': '[128,128,128,128]', 'learning_rate': 0.005, 'message_dropout': 0.0, 'node_dropout': 0.1, 'reg_weight': 1e-05}
#  10%|█         | 20/200 [40:16:58<312:36:10, 6252.06s/trial, best loss: -0.1437]C:\Users\s8347434\Documents\RecBole-GNN\.venv\Lib\site-packages\torch_geometric\deprecation.py:26: UserWarning:
#
# 'dropout_adj' is deprecated, use 'dropout_edge' instead