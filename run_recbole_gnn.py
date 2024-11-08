import argparse

from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='ForwardGNN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='test.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

# SASRec 0.2385
# LightGCN 0.1857
# BPR 0.1537
# UserKNN 0.
# ItemKNN 0.1974


# On Random Split!
### ForwardGNN, 20 Epochs, train loss: 1.0954
# 'precision@10': 0.2989, 'hit@10': 0.9417, 'mrr@10': 0.5931, 'ndcg@10': 0.374,
# 'map@10': 0.2264, 'itemcoverage@10': 0.5633, 'averagepopularity@10': 160.2382, 'tailpercentage@10': 0.0031})

### LightGCN, 20 Epochs, train loss: 2.3252
# 'precision@10': 0.3989, 'hit@10': 0.9788, 'mrr@10': 0.7279, 'ndcg@10': 0.496,
# 'map@10': 0.3393, 'itemcoverage@10': 0.3684, 'averagepopularity@10': 202.8382, 'tailpercentage@10': 0.0})

### LightGCN goes Fordward-Forward, train loss: 1.0915
# 'precision@10': 0.3922, 'hit@10': 0.9724, 'mrr@10': 0.722, 'ndcg@10': 0.4895,
# 'map@10': 0.334, 'itemcoverage@10': 0.3844, 'averagepopularity@10': 219.1405, 'tailpercentage@10': 0.0034})