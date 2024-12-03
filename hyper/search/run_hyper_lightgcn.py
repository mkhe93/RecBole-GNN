import argparse
import torch

from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='../parameter_fixed/hyper_lightgcn_djc100000.yaml', help='fixed config files')
    parser.add_argument('--params_file', type=str, default='../parameter_space/params_lightgcn.hyper', help='parameters file')
    parser.add_argument('--output_file', type=str, default='../results/lightgcn-hyper-djc100000.result', help='output file')
    args, _ = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set. Others: 'bayes','random'
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hp = HyperTuning(objective_function, algo='bayes', max_evals=200, early_stop=200,
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    torch.set_num_threads(8)
    hp.run()
    hp.export_result(output_file=args.output_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])

if __name__ == '__main__':
    main()
