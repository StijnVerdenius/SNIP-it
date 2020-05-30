import argparse
import sys
import time

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import GeneralModel
from models.statistics import Metrics
from models.statistics.Flops import FLOPCounter
from models.statistics.Saliency import Saliency
from utils.model_utils import find_right_model
from utils.system_utils import *


class DefaultTrainer:

    """
    Standard CV classification with pruning
    """

    def __init__(self,
                 model: GeneralModel,
                 loss: GeneralModel,
                 optimizer: Optimizer,
                 device,
                 arguments: argparse.Namespace,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 metrics: Metrics,
                 criterion: GeneralModel
                 ):

        self._test_loader = test_loader
        self._train_loader = train_loader
        self._loss_function = loss
        self._model = model
        self._arguments = arguments
        self._optimizer = optimizer
        self._device = device
        self._global_steps = 0
        self.out = metrics.log_line
        DATA_MANAGER.set_date_stamp(addition=arguments.run_name)
        self._writer = SummaryWriter(os.path.join(DATA_MANAGER.directory, RESULTS_DIR, DATA_MANAGER.stamp, SUMMARY_DIR))
        self._metrics: Metrics = metrics
        self._metrics.init_training(self._writer)
        self._acc_buffer = []
        self._loss_buffer = []
        self._elapsed_buffer = []
        self._criterion = criterion

        self.ts = None

        batch = next(iter(self._test_loader))
        self.saliency = Saliency(model, device, batch[0][:8])
        self._metrics.write_arguments(arguments)
        self._flopcounter = FLOPCounter(model, batch[0][:8], self._arguments.batch_size, device=device)
        self._metrics.model_to_tensorboard(model, timestep=-1)

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True):
        """ one iteration of forward-backward """

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)

        # update metrics
        self._metrics.update_batch(train)

        # record time
        if "cuda" in str(self._device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # forward pass
        accuracy, loss, out = self._forward_pass(x, y, train=train)

        # backward pass
        if train:
            self._backward_pass(loss)

        # record time
        if "cuda" in str(self._device):
            end.record()
            torch.cuda.synchronize(self._device)
            time = start.elapsed_time(end)
        else:
            time = 0

        # free memory
        for tens in [out, y, x, loss]:
            tens.detach()

        return accuracy, loss.item(), time

    def _forward_pass(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      train: bool = True):
        """ implementation of a forward pass """

        if train:
            self._optimizer.zero_grad()
            if self._model.is_maskable:
                self._model.apply_weight_mask()

        out = self._model(x).squeeze()
        loss = self._loss_function(
            output=out,
            target=y,
            weight_generator=self._model.parameters(),
            model=self._model,
            criterion=self._criterion
        )
        accuracy = self._get_accuracy(out, y)
        return accuracy, loss, out

    def _backward_pass(self, loss):
        """ implementation of a backward pass """

        loss.backward()
        self._model.insert_noise_for_gradient(self._arguments.grad_noise)
        if self._arguments.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._arguments.grad_clip)
        self._optimizer.step()
        if self._model.is_maskable:
            self._model.apply_weight_mask()

    def _epoch_iteration(self):
        """ implementation of an epoch """

        self.out("\n")

        self._acc_buffer, self._loss_buffer = self._metrics.update_epoch()

        for batch_num, batch in enumerate(self._train_loader):
            self.out(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            if self._model.is_tracking_weights:
                self._model.save_prev_weights()

            acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)

            if self._model.is_tracking_weights:
                self._model.update_tracked_weights(self._metrics.batch_train)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)
            self._elapsed_buffer.append(elapsed)

            self._log(batch_num)

            self._check_exit_conditions_epoch_iteration()

        self.out("\n")

    def _log(self, batch_num: int):
        """ logs to terminal and tensorboard if the time is right"""

        if (batch_num % self._arguments.eval_freq) == 0:
            # validate on test and train set
            train_acc, train_loss = np.mean(self._acc_buffer), np.mean(self._loss_buffer)
            test_acc, test_loss, test_elapsed = self.validate()
            self._elapsed_buffer += test_elapsed

            # log metrics
            self._add_metrics(test_acc, test_loss, train_acc, train_loss)

            # reset for next log
            self._acc_buffer, self._loss_buffer, self._elapsed_buffer = [], [], []

            # print to terminal
            self.out(self._metrics.printable_last)

    def validate(self):
        """ validates the model on test set """

        self.out("\n")

        # init test mode
        self._model.eval()
        cum_acc, cum_loss, cum_elapsed = [], [], []

        with torch.no_grad():
            for batch_num, batch in enumerate(self._test_loader):
                acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)
                cum_acc.append(acc)
                cum_loss.append(loss),
                cum_elapsed.append(elapsed)
                self.out(f"\rEvaluating... {batch_num}/{len(self._test_loader)}", end='')
        self.out("\n")

        # put back into train mode
        self._model.train()

        return float(np.mean(cum_acc)), float(np.mean(cum_loss)), cum_elapsed

    def _add_metrics(self, test_acc, test_loss, train_acc, train_loss):
        """
        save metrics
        """

        sparsity = self._model.pruned_percentage
        spasity_index = 2 * ((sparsity * test_acc) / (1e-8 + sparsity + test_acc))

        flops_per_sample, total_seen = self._flopcounter.count_flops(self._metrics.batch_train)

        self._metrics.add(train_acc, key="acc/train")
        self._metrics.add(train_loss, key="loss/train")
        self._metrics.add(test_loss, key="loss/test")
        self._metrics.add(test_acc, key="acc/test")
        self._metrics.add(sparsity, key="sparse/weight")
        self._metrics.add(self._model.structural_sparsity, key="sparse/node")
        self._metrics.add(spasity_index, key="sparse/hm")
        self._metrics.add(np.log(self._model.compressed_size), key="sparse/log_disk_size")
        self._metrics.add(np.mean(self._elapsed_buffer), key="time/gpu_time")
        self._metrics.add(int(flops_per_sample), key="time/flops_per_sample")
        self._metrics.add(np.log10(total_seen), key="time/flops_log_cum")
        if torch.cuda.is_available():
            self._metrics.add(torch.cuda.memory_allocated(0), key="cuda/ram_footprint")
        self._metrics.timeit()

    def train(self):
        """ main training function """

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self._arguments)
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "calling_command.txt"), str(" ".join(sys.argv)))

        # data gathering
        epoch = self._metrics._epoch

        self._model.train()

        try:

            self.out(
                f"{PRINTCOLOR_BOLD}Started training{PRINTCOLOR_END}"
            )

            if self._arguments.skip_first_plot:
                self._metrics.handle_weight_plotting(0, trainer_ns=self)

            # if snip we prune before training
            if self._arguments.prune_criterion in SINGLE_SHOT:
                self._criterion.prune(self._arguments.pruning_limit,
                                      train_loader=self._train_loader,
                                      manager=DATA_MANAGER)
                if self._arguments.prune_criterion in STRUCTURED_SINGLE_SHOT:
                    self._optimizer = find_right_model(OPTIMS, self._arguments.optimizer,
                                                       params=self._model.parameters(),
                                                       lr=self._arguments.learning_rate,
                                                       weight_decay=self._arguments.l2_reg)
                    self._metrics.model_to_tensorboard(self._model, timestep=epoch)

            # do training
            for epoch in range(epoch, self._arguments.epochs + epoch):
                self.out(f"\n\n{PRINTCOLOR_BOLD}EPOCH {epoch} {PRINTCOLOR_END} \n\n")

                # do epoch
                self._epoch_iteration()

                # plotting
                if (epoch % self._arguments.plot_weights_freq) == 0 and self._arguments.plot_weights_freq > 0:
                    self._metrics.handle_weight_plotting(epoch, trainer_ns=self)

                # do all related to pruning
                self._handle_pruning(epoch)

                # save what needs to be saved
                self._handle_backing_up(epoch)

            if self._arguments.skip_first_plot:
                self._metrics.handle_weight_plotting(epoch + 1, trainer_ns=self)

            # example last save
            save_models([self._model, self._metrics], "finished")

        except KeyboardInterrupt as e:
            self.out(f"Killed by user: {e} at {time.time()}")
            save_models([self._model, self._metrics], f"KILLED_at_epoch_{epoch}")
            sys.stdout.flush()
            DATA_MANAGER.write_to_file(
                os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
            self._writer.close()
            exit(69)
        except Exception as e:
            self._writer.close()
            report_error(e, self._model, epoch, self._metrics)

        # flush prints
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
        self._writer.close()

    def _handle_backing_up(self, epoch):
        if (epoch % self._arguments.save_freq) == 0 and epoch > 0:
            self.out("\nSAVING...\n")
            save_models(
                [self._model, self._metrics],
                f"save_at_epoch_{epoch}"
            )
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"),
            self._metrics.log
        )

    def _handle_pruning(self, epoch):
        if self._is_pruning_time(epoch):
            if self._is_not_finished_pruning():
                self.out("\nPRUNING...\n")
                self._criterion.prune(
                    percentage=self._arguments.pruning_rate,
                    train_loader=self._train_loader,
                    manager=DATA_MANAGER
                )
                if self._arguments.prune_criterion in DURING_TRAINING:
                    self._optimizer = find_right_model(
                        OPTIMS, self._arguments.optimizer,
                        params=self._model.parameters(),
                        lr=self._arguments.learning_rate,
                        weight_decay=self._arguments.l2_reg
                    )
                    self._metrics.model_to_tensorboard(self._model, timestep=epoch)
                if self._model.is_rewindable:
                    self.out("rewinding weights to checkpoint...\n")
                    self._model.do_rewind()
            if self._model.is_growable:
                self.out("growing too...\n")
                self._criterion.grow(self._arguments.growing_rate)

        if self._is_checkpoint_time(epoch):
            self.out(f"\nCreating weights checkpoint at epoch {epoch}\n")
            self._model.save_rewind_weights()

    def _is_not_finished_pruning(self):
        return self._arguments.pruning_limit > self._model.pruned_percentage \
               or \
               (
                       self._arguments.prune_criterion in DURING_TRAINING
                       and
                       self._arguments.pruning_limit > self._model.structural_sparsity
               )

    @staticmethod
    def _get_accuracy(output, y):
        predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
        correct = y.eq(predictions).sum().item()
        return correct / output.shape[0]

    def _is_checkpoint_time(self, epoch: int):
        return epoch == self._arguments.rewind_to and self._model.is_rewindable

    def _is_pruning_time(self, epoch: int):
        if self._arguments.prune_criterion == "EmptyCrit":
            return False
        epoch -= self._arguments.prune_delay
        return (epoch % self._arguments.prune_freq) == 0 and \
               epoch > 0 and \
               self._model.is_maskable and \
               self._arguments.prune_criterion not in SINGLE_SHOT

    def _check_exit_conditions_epoch_iteration(self, patience=1):

        time_passed = datetime.now() - DATA_MANAGER.actual_date
        # check if runtime is expired
        if (time_passed.total_seconds() > (self._arguments.max_training_minutes * 60)) \
                and \
                self._arguments.max_training_minutes > 0:
            raise KeyboardInterrupt(
                f"Process killed because {self._arguments.max_training_minutes} minutes passed "
                f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")
        if patience == 0:
            raise NotImplementedError("feature to implement",
                                      KeyboardInterrupt("Process killed because patience is zero"))
