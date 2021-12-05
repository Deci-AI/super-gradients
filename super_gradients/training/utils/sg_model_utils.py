import os
import sys
import socket
import time
from multiprocessing import Process
from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.tensorboard import SummaryWriter

from super_gradients.training.exceptions.dataset_exceptions import UnsupportedBatchItemsFormat

# TODO: These utils should move to sg_model package as internal (private) helper functions


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
