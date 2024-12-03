from time import time
import math
import os
import torch
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from tqdm import tqdm
from recbole.trainer import Trainer, PretrainTrainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)


class ForwardGNNTrainer(PretrainTrainer):
    r"""ForwardTrainer is designed to train Forward-Forward models.
    """
    def __init__(self, config, model):
        super(ForwardGNNTrainer, self).__init__(config, model)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if self.model.train_stage == "pretrain":
            best_valid_score, best_valid_result = self.pretrain(train_data, valid_data, saved, verbose, show_progress)

            self._add_hparam_to_tensorboard(self.best_valid_score)

            return best_valid_score, best_valid_result
        elif self.model.train_stage == "finetune":
            return self.finetune(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError(
                "Please make sure that the 'train_stage' is 'pretrain' or 'finetune'!"
            )

    def pretrain(self, train_data, valid_data=None, saved=True, verbose=True, show_progress=False):

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)  # needed for metrics such as ItemCoverage

        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)

        valid_step = 0
        # FIXME: iterate here over the different layers
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch_layerwise(
                train_data, epoch_idx, verbose=verbose, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

            # save pretrained model
            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"], str(epoch_idx + 1)
                    ),
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = (
                        set_color("Saving current", "blue") + ": %s" % saved_model_file
                )
                if verbose:
                    self.logger.info(update_output)

        # add best valid_score, only if validation has been done
        return self.best_valid_score, self.best_valid_result

    def finetune(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch_finetuning(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        return self.best_valid_score, self.best_valid_result

    def _train_epoch_finetuning(self, train_data, epoch_idx, loss_func=None, show_progress=False, parameters=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()

        params = [param for name, param in self.model.named_parameters() if "final_aggregation" in name]

        self.optimizer = self._build_optimizer(params=params)
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            loss = loss + sync_loss
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
        return total_loss

    def _train_epoch_layerwise(self, train_data, epoch_idx, loss_func=None, verbose=False, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        total_loss = None # FIXME: where to inizialize the total_loss?

        for i, layer in enumerate(self.model.forward_convs):

            for layer_epoch_idx in range(0, 1):
                # train each layer for certain amount of epochs
                training_start_time = time()

                layer_desc = f"Layer {i + 1}/{self.config["n_layers"]}"

                for name, param in self.model.named_parameters():
                    if (
                            "user_embedding.weight" in name
                            or "item_embedding.weight" in name
                            or f"forward_convs.{i}" in name
                    ):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                params = [
                    param
                    for name, param in self.model.named_parameters()
                    if (
                        "item_embedding.weight" in name
                        or "user_embedding.weight" in name
                        or f"forward_convs.{i}" in name
                    )
                ]
                #self.layer_optimizer = self._build_optimizer(params=params)

                iter_data = (
                    tqdm(
                        train_data,
                        total=len(train_data),
                        ncols=100,
                        desc=set_color(f"Train {epoch_idx:>5} | {layer_desc}", "pink"),
                    )
                    if show_progress
                    else train_data
                )

                if not self.config["single_spec"] and train_data.shuffle:
                    train_data.sampler.set_epoch(epoch_idx)

                for batch_idx, interaction in enumerate(iter_data):
                    interaction = interaction.to(self.device)

                    #self.layer_optimizer.zero_grad()
                    #self.optimizer.zero_grad()
                    sync_loss = 0
                    #if not self.config["single_spec"]:
                    #    self.set_reduce_hook()
                    #    sync_loss = self.sync_grad_loss()

                    with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                        if layer.requires_training:
                            embeddings, user, pos_item, neg_items = self.model.get_layer_train_data(interaction)
                            losses, logits = layer.forward_train(embeddings, user, pos_item, neg_items, self.model.edge_index, self.model.edge_weight, self.config["theta"])
                        else:
                            continue

                    if show_progress:
                        if logits:
                            separation = logits[0] - logits[1]
                            iter_data.set_description(
                                f"[Layer {i+1}: Epoch-{epoch_idx}] | "
                                f"Goodness: pos={logits[0]:.4f}, neg={logits[1]:.4f} | "
                                f"Separation: {separation:.4f} | "
                            )

                    if isinstance(losses, tuple):
                        loss = sum(losses)
                        loss_tuple = tuple(per_loss.item() for per_loss in losses)
                        total_loss = (
                            loss_tuple
                            if total_loss is None
                            else tuple(map(sum, zip(total_loss, loss_tuple)))
                        )
                    else:
                        loss = losses
                        total_loss = (
                            losses.item() if total_loss is None else total_loss + losses.item()
                        )
                    self._check_nan(loss)
                    loss = loss + sync_loss
                    #loss.backward()
                    if self.clip_grad_norm:
                        clip_grad_norm_(params, **self.clip_grad_norm)

                    #self.layer_optimizer.step()
                    #self.optimizer.step()

                    if self.gpu_available and show_progress:
                        iter_data.set_postfix_str(
                            set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                        )


                    # For layer-wise training
                    #self.train_loss_dict[layer_epoch_idx] = (
                    #    sum(total_loss) if isinstance(total_loss, tuple) else total_loss
                    #)
                    training_end_time = time()
                    #train_loss_output = self._generate_train_loss_output(
                    #    layer_epoch_idx, training_start_time, training_end_time, total_loss
                    #)
                    #if verbose:
                    #    self.logger.info(train_loss_output)
                    #self._add_train_loss_to_tensorboard(layer_epoch_idx, total_loss)


        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress
        )
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            if self.config["layer_evaluation"]:
                eval_func = self._full_sort_batch_eval_per_layer
            else:
                eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        num_sample = 0
        if self.config["layer_evaluation"]:
            result = []

            for layer_num in range(self.config["n_layers"]):
                layer_desc = f"layer {layer_num + 1}/{self.config["n_layers"]}"
                iter_data = (
                    tqdm(
                        eval_data,
                        total=len(eval_data),
                        ncols=100,
                        desc=set_color(f"Evaluate {layer_desc}  ", "pink"),
                    )
                    if show_progress
                    else eval_data
                )

                for batch_idx, batched_data in enumerate(iter_data):
                    num_sample += len(batched_data)
                    interaction, scores, positive_u, positive_i = eval_func(batched_data, layer_num)
                    if self.gpu_available and show_progress:
                        iter_data.set_postfix_str(
                            set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                        )
                    self.eval_collector.eval_batch_collect(
                        scores, interaction, positive_u, positive_i
                    )
                self.eval_collector.model_collect(self.model)
                struct = self.eval_collector.get_data_struct()
                result_tmp = self.evaluator.evaluate(struct)
                if not self.config["single_spec"]:
                    result = self._map_reduce(result_tmp, num_sample)
                self.wandblogger.log_eval_metrics(result_tmp, head="eval")
                result.append(result_tmp)
        else:
            iter_data = (
                tqdm(
                    eval_data,
                    total=len(eval_data),
                    ncols=100,
                    desc=set_color(f"Evaluate   ", "pink"),
                )
                if show_progress
                else eval_data
            )

            for batch_idx, batched_data in enumerate(iter_data):
                num_sample += len(batched_data)
                interaction, scores, positive_u, positive_i = eval_func(batched_data)
                if self.gpu_available and show_progress:
                    iter_data.set_postfix_str(
                        set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                    )
                self.eval_collector.eval_batch_collect(
                    scores, interaction, positive_u, positive_i
                )
            self.eval_collector.model_collect(self.model)
            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)
            if not self.config["single_spec"]:
                result = self._map_reduce(result, num_sample)
            self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    def _full_sort_batch_eval_per_layer(self, batched_data, layer_num):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict_per_layer(interaction.to(self.device), layer_num)
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict_per_layer(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

class NCLTrainer(Trainer):
    def __init__(self, config, model):
        super(NCLTrainer, self).__init__(config, model)

        self.num_m_step = config['m_step']
        assert self.num_m_step is not None

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):

            # only differences from the original trainer
            if epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                if epoch_idx < self.config['warm_up_step']:
                    losses = losses[:-1]
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss


class HMLETTrainer(Trainer):
    def __init__(self, config, model):
        super(HMLETTrainer, self).__init__(config, model)

        self.warm_up_epochs = config['warm_up_epochs']
        self.ori_temp = config['ori_temp']
        self.min_temp = config['min_temp']
        self.gum_temp_decay = config['gum_temp_decay']
        self.epoch_temp_decay = config['epoch_temp_decay']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx > self.warm_up_epochs:
            # Temp decay
            gum_temp = self.ori_temp * math.exp(-self.gum_temp_decay*(epoch_idx - self.warm_up_epochs))
            self.model.gum_temp = max(gum_temp, self.min_temp)
            self.logger.info(f'Current gumbel softmax temperature: {self.model.gum_temp}')

            for gating in self.model.gating_nets:
                self.model._gating_freeze(gating, True)
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)


class SEPTTrainer(Trainer):
    def __init__(self, config, model):
        super(SEPTTrainer, self).__init__(config, model)
        self.warm_up_epochs = config['warm_up_epochs']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx < self.warm_up_epochs:
            loss_func = self.model.calculate_rec_loss
        else:
            self.model.subgraph_construction()
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)