# Changes in the backend of RecBole

- AbstractTrainer, L206: skip nan loss
- hypertuning.py: L117 : rng = np.random.default_rng(seed)
- trainer.py: user_evaluation
- evaluator.py: user_evaluation
- base_metric.py: user_evaluation
- dataset.py: average_popularity_user, actions_of_items and actions_of_users

TODO: in als.py -> error with blas beheben