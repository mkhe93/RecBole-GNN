import argparse
import os
import torch

from recbole_gnn.quick_start import run_recbole_gnn

if __name__ == '__main__':
    model = 'SGL'
    dataset = 'real-life-atomic-100000'
    #dataset = 'ml-100k'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=model, help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default=dataset, help='name of datasets')
    parser.add_argument('--config_files', type=str, default='test.yaml', help='config files')

    layer_aggregation = 'mean'
    final_aggregation = 'mean'

    torch.set_num_threads(8)

    args, _ = parser.parse_known_args()

    if final_aggregation not in ['softmax', 'attention']:
        config_dict = {
            'train_stage': 'pretrain',
            'layer_aggregation': layer_aggregation,
            'final_aggregation': final_aggregation
        }
        config_file_list = args.config_files.strip().split(' ') if args.config_files else None
        run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict, saved=False)

    else:
        pre_train_epochs = 5
        save_step = 5

        config_dict = {
            'train_stage': 'pretrain',
            'layer_aggregation': layer_aggregation,
            'final_aggregation': final_aggregation,
            'pretrain_epochs': pre_train_epochs,
            'save_step': save_step
        }
        config_file_list = args.config_files.strip().split(' ') if args.config_files else None
        run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list,
                        config_dict=config_dict, saved=False)

        config_dict = {
            'train_stage': 'finetune',
            'epochs': 5,
            'pre_model_path': f'./saved/{model}-{dataset}-{pre_train_epochs}.pth'
        }
        run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list,
                        config_dict=config_dict)

## 16 Jan 08:13
# 'precision@10': 0.0284, 'hit@10': 0.2225, 'mrr@10': 0.0948, 'ndcg@10': 0.0608,
# 'map@10': 0.0332, 'itemcoverage@10': 0.1451, 'averagepopularity@10': 44.1465, 'tailpercentage@10': 0.0037})