import os
import sys
import socket
import time
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Tuple, Union, Dict
import random

from super_gradients.common.abstractions.abstract_logger import get_logger
from deprecate import deprecated
from treelib import Tree
from termcolor import colored
import torch
from torch.utils.tensorboard import SummaryWriter

from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training.exceptions.dataset_exceptions import UnsupportedBatchItemsFormat


# TODO: These utils should move to sg_trainer package as internal (private) helper functions

IS_BETTER_COLOR = {True: "green", False: "red"}
IS_GREATER_SYMBOLS = {True: "↗", False: "↘"}

logger = get_logger(__name__)

@dataclass
class MonitoredValue:
    """Store a value and some indicators relative to its past iterations.

    The value can be a metric/loss, and the iteration can be epochs/batch.
    """
    current: float = None
    previous: float = None
    best: float = None
    change_from_previous: float = None
    change_from_best: float = None
    is_better_than_previous: bool = None
    is_best_value: bool = None


def update_monitored_value(previous_monitored_value: MonitoredValue, new_value: float,
                           greater_is_better: bool) -> MonitoredValue:
    """Update the given ValueToMonitor object (could be a loss or a metric) with the new value

    :param previous_monitored_value: The stats about the value that is monitored throughout epochs.
    :param new_value: The value of the current epoch that will be used to update previous_monitored_value
    :param greater_is_better: True when a greater value means better result.
    :return:
    """
    previous_value, previous_best_value = previous_monitored_value.current, previous_monitored_value.best

    if previous_best_value is None:
        previous_best_value = previous_value
    elif greater_is_better:
        previous_best_value = max(previous_value, previous_best_value)
    else:
        previous_best_value = min(previous_value, previous_best_value)

    if previous_value is None:
        change_from_previous = None
        change_from_best = None
        is_better_than_previous = None
        is_best_value = None
    else:
        change_from_previous = new_value - previous_value
        change_from_best = new_value - previous_best_value
        is_better_than_previous = change_from_previous >= 0 if greater_is_better else change_from_previous <= 0
        is_best_value = change_from_best >= 0 if greater_is_better else change_from_best <= 0

    return MonitoredValue(current=new_value, previous=previous_value, best=previous_best_value,
                          change_from_previous=change_from_previous, change_from_best=change_from_best,
                          is_better_than_previous=is_better_than_previous, is_best_value=is_best_value)


def update_monitored_values_dict(monitored_values_dict: Dict[str, MonitoredValue],
                                 new_values_dict: Dict[str, float]) -> Dict[str, MonitoredValue]:
    """Update the given ValueToMonitor object (could be a loss or a metric) with the new value

    :param monitored_values_dict: Dict mapping value names to their stats throughout epochs.
    :param new_values_dict: Dict mapping value names to their new (i.e. current epoch) value.
    :return: Updated monitored_values_dict
    """
    for monitored_value_name in monitored_values_dict.keys():
        monitored_values_dict[monitored_value_name] = update_monitored_value(
            new_value=new_values_dict[monitored_value_name],
            previous_monitored_value=monitored_values_dict[monitored_value_name],
            greater_is_better=False
        )
    return monitored_values_dict


def display_epoch_summary(epoch: int, n_digits: int,
                          train_monitored_values: Dict[str, MonitoredValue],
                          valid_monitored_values: Dict[str, MonitoredValue]) -> None:
    """Display a summary of loss/metric of interest, for a given epoch.

        :param epoch: the number of epoch.
        :param n_digits: number of digits to display on screen for float values
        :param train_monitored_values: mapping of loss/metric with their stats that will be displayed
        :param valid_monitored_values: mapping of loss/metric with their stats that will be displayed
        :return:
    """

    def _format_to_str(val: float) -> str:
        return str(round(val, n_digits))

    def _generate_tree(value_name: str, monitored_value: MonitoredValue) -> Tree:
        """Generate a tree that represents the stats of a given loss/metric."""

        current = _format_to_str(monitored_value.current)
        root_id = str(hash(f"{value_name} = {current}")) + str(random.random())

        tree = Tree()
        tree.create_node(tag=f"{value_name.capitalize()} = {current}", identifier=root_id)

        if monitored_value.previous is not None:
            previous = _format_to_str(monitored_value.previous)
            best = _format_to_str(monitored_value.best)
            change_from_previous = _format_to_str(monitored_value.change_from_previous)
            change_from_best = _format_to_str(monitored_value.change_from_best)

            diff_with_prev_colored = colored(
                text=f"{IS_GREATER_SYMBOLS[monitored_value.change_from_previous > 0]} {change_from_previous}",
                color=IS_BETTER_COLOR[monitored_value.is_better_than_previous]
            )
            diff_with_best_colored = colored(
                text=f"{IS_GREATER_SYMBOLS[monitored_value.change_from_best > 0]} {change_from_best}",
                color=IS_BETTER_COLOR[monitored_value.is_best_value]
            )

            tree.create_node(
                tag=f"Epoch N-1      = {previous:6} ({diff_with_prev_colored:8})",
                identifier=f"0_previous_{root_id}",
                parent=root_id
            )
            tree.create_node(
                tag=f"Best until now = {best:6} ({diff_with_best_colored:8})",
                identifier=f"1_best_{root_id}",
                parent=root_id
            )
        return tree

    train_tree = Tree()
    train_tree.create_node("Training", "Training")
    for name, value in train_monitored_values.items():
        train_tree.paste('Training', new_tree=_generate_tree(name, monitored_value=value))

    valid_tree = Tree()
    valid_tree.create_node("Validation", "Validation")
    for name, value in valid_monitored_values.items():
        valid_tree.paste('Validation', new_tree=_generate_tree(name, monitored_value=value))

    summary_tree = Tree()
    summary_tree.create_node(f"SUMMARY OF EPOCH {epoch}", "Summary")
    summary_tree.paste("Summary", train_tree)
    summary_tree.paste("Summary", valid_tree)
    summary_tree.show()


def try_port(port):
    """
    try_port - Helper method for tensorboard port binding
    :param port:
    :return:
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_port_available = False
    try:
        sock.bind(("localhost", port))
        is_port_available = True

    except Exception as ex:
        print('Port ' + str(port) + ' is in use' + str(ex))

    sock.close()
    return is_port_available


def launch_tensorboard_process(checkpoints_dir_path: str, sleep_postpone: bool = True, port: int = None) -> Tuple[Process, int]:
    """
    launch_tensorboard_process - Default behavior is to scan all free ports from 6006-6016 and try using them
                                 unless port is defined by the user
        :param checkpoints_dir_path:
        :param sleep_postpone:
        :param port:
        :return: tuple of tb process, port
    """
    logdir_path = str(Path(checkpoints_dir_path).parent.absolute())
    tb_cmd = 'tensorboard --logdir=' + logdir_path + ' --bind_all'
    if port is not None:
        tb_ports = [port]
    else:
        tb_ports = range(6006, 6016)

    for tb_port in tb_ports:
        if not try_port(tb_port):
            continue
        else:
            print('Starting Tensor-Board process on port: ' + str(tb_port))
            tensor_board_process = Process(target=os.system, args=([tb_cmd + ' --port=' + str(tb_port)]))
            tensor_board_process.daemon = True
            tensor_board_process.start()

            # LET THE TENSORBOARD PROCESS START
            if sleep_postpone:
                time.sleep(3)
            return tensor_board_process, tb_port

    # RETURNING IRRELEVANT VALUES
    print('Failed to initialize Tensor-Board process on port: ' + ', '.join(map(str, tb_ports)))
    return None, -1


def init_summary_writer(tb_dir, checkpoint_loaded, user_prompt=False):
    """Remove previous tensorboard files from directory and launch a tensor board process"""
    # If the training is from scratch, Walk through destination folder and delete existing tensorboard logs
    user = ''
    if not checkpoint_loaded:
        for filename in os.listdir(tb_dir):
            if 'events' in filename:
                if not user_prompt:
                    print('"{}" will not be deleted'.format(filename))
                    continue

                while True:
                    # Verify with user before deleting old tensorboard files
                    user = input('\nOLDER TENSORBOARD FILES EXISTS IN EXPERIMENT FOLDER:\n"{}"\n'
                                 'DO YOU WANT TO DELETE THEM? [y/n]'
                                 .format(filename)) if (user != 'n' or user != 'y') else user
                    if user == 'y':
                        os.remove('{}/{}'.format(tb_dir, filename))
                        print('DELETED: {}!'.format(filename))
                        break
                    elif user == 'n':
                        print('"{}" will not be deleted'.format(filename))
                        break
                    print('Unknown answer...')

    # Launch a tensorboard process
    return SummaryWriter(tb_dir)


def add_log_to_file(filename, results_titles_list, results_values_list, epoch, max_epochs):
    """Add a message to the log file"""
    # -Note: opening and closing the file every time is in-efficient. It is done for experimental purposes
    with open(filename, 'a') as f:
        f.write('\nEpoch (%d/%d)  - ' % (epoch, max_epochs))
        for result_title, result_value in zip(results_titles_list, results_values_list):
            if isinstance(result_value, torch.Tensor):
                result_value = result_value.item()
            f.write(result_title + ': ' + str(result_value) + '\t')


def write_training_results(writer, results_titles_list, results_values_list, epoch):
    """Stores the training and validation loss and accuracy for current epoch in a tensorboard file"""
    for res_key, res_val in zip(results_titles_list, results_values_list):
        # USE ONLY LOWER-CASE LETTERS AND REPLACE SPACES WITH '_' TO AVOID MANY TITLES FOR THE SAME KEY
        corrected_res_key = res_key.lower().replace(' ', '_')
        writer.add_scalar(corrected_res_key, res_val, epoch)
    writer.flush()


def write_hpms(writer, hpmstructs=[], special_conf={}):
    """Stores the training and dataset hyper params in the tensorboard file"""
    hpm_string = ""
    for hpm in hpmstructs:
        for key, val in hpm.__dict__.items():
            hpm_string += '{}: {}  \n  '.format(key, val)
    for key, val in special_conf.items():
        hpm_string += '{}: {}  \n  '.format(key, val)
    writer.add_text("Hyper_parameters", hpm_string)
    writer.flush()


# TODO: This should probably move into datasets/datasets_utils.py?
def unpack_batch_items(batch_items: Union[tuple, torch.Tensor]):
    """
    Adds support for unpacking batch items in train/validation loop.

    @param batch_items: (Union[tuple, torch.Tensor]) returned by the data loader, which is expected to be in one of
         the following formats:
            1. torch.Tensor or tuple, s.t inputs = batch_items[0], targets = batch_items[1] and len(batch_items) = 2
            2. tuple: (inputs, targets, additional_batch_items)

         where inputs are fed to the network, targets are their corresponding labels and additional_batch_items is a
         dictionary (format {additional_batch_item_i_name: additional_batch_item_i ...}) which can be accessed through
         the phase context under the attribute additional_batch_item_i_name, using a phase callback.


    @return: inputs, target, additional_batch_items
    """
    additional_batch_items = {}
    if len(batch_items) == 2:
        inputs, target = batch_items

    elif len(batch_items) == 3:
        inputs, target, additional_batch_items = batch_items

    else:
        raise UnsupportedBatchItemsFormat()

    return inputs, target, additional_batch_items


def log_uncaught_exceptions(logger):
    """
    Makes logger log uncaught exceptions
    @param logger: logging.Logger

    @return: None
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

@deprecated(deprecated_in='2.1.0', remove_in='2.3.0')
def scale_params_for_yolov5(cfg):
    """
    Scale:
        * learning rate,
        * weight decay,
        * box_loss_gain,
        * cls_loss_gain,
        * obj_loss_gain
    according to:
        * effective batch size
        * DDP world size
        * image size
        * num YOLO output layers
        * num classes
    """

    # Scale LR and weight decay
    is_ddp = cfg.trainer .multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if is_ddp else 1

    # Scale LR and WD for DDP due to gradients being averaged between devices
    # Equivalent to loss * WORLD_SIZE in ultralytics
    cfg.training_hyperparams.initial_lr *= world_size
    cfg.training_hyperparams.warmup_bias_lr *= world_size
    cfg.training_hyperparams.optimizer_params.weight_decay /= world_size

    # Scale WD with a factor of [effective batch size]/64.
    batch_size, batch_accumulate = cfg.dataset_params.batch_size, cfg.training_hyperparams.batch_accumulate
    batch_size_factor = cfg.trainer .num_devices if is_ddp else cfg.trainer .dataset_interface.batch_size_factor
    effective_batch_size = batch_size * batch_size_factor * batch_accumulate
    cfg.training_hyperparams.optimizer_params.weight_decay *= effective_batch_size / 64.

    # Scale EMA beta to match Ultralytics update
    cfg.training_hyperparams.ema_params.beta = cfg.training_hyperparams.max_epochs * len(cfg.trainer .train_loader) / 2000.

    log_msg = \
        f"""
        IMPORTANT:\n
        Training with world size of {world_size}, {'DDP' if is_ddp else 'no DDP'}, effective batch size of {effective_batch_size},
        scaled:
            * initial_lr to {cfg.training_hyperparams.initial_lr};
            * warmup_bias_lr to {cfg.training_hyperparams.warmup_bias_lr};
            * weight_decay to {cfg.training_hyperparams.optimizer_params.weight_decay};
            * EMA beta to {cfg.training_hyperparams.ema_params.beta};
        """

    if cfg.training_hyperparams.loss == 'yolo_v5_loss':
        # Scale loss gains
        model = cfg.trainer .net
        model = model.module if hasattr(model, 'module') else model
        num_levels = model._head._modules_list[-1].detection_layers_num
        train_image_size = cfg.dataset_params.train_image_size

        num_branches_norm = 3. / num_levels
        num_classes_norm = len(cfg.trainer .classes) / 80.
        image_size_norm = train_image_size / 640.
        cfg.training_hyperparams.criterion_params.box_loss_gain *= num_branches_norm
        cfg.training_hyperparams.criterion_params.cls_loss_gain *= num_classes_norm * num_branches_norm
        cfg.training_hyperparams.criterion_params.obj_loss_gain *= image_size_norm ** 2 * num_branches_norm

        log_msg += \
            f"""
            * box_loss_gain to {cfg.training_hyperparams.criterion_params.box_loss_gain};
            * cls_loss_gain to {cfg.training_hyperparams.criterion_params.cls_loss_gain};
            * obj_loss_gain to {cfg.training_hyperparams.criterion_params.obj_loss_gain};
            """

    logger.info(log_msg)
    return cfg