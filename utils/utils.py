"""
    Comments:   This is the utility file for running the main program
    ToDo:       * Make it into a class for better usuability
    **********************************************************************************
"""
import numpy as np
import pandas as pd
import os
import json
import shutil
from logging import StreamHandler, INFO, DEBUG, Formatter, FileHandler, getLogger
import pdb


def dir_maker(path, clean=False):
    """Creating folder structures based on the path specified.

    """

    # First check if the directory exists
    if path.exists():
        print("Path already exists.")
        if clean:
            print("Cleaning")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    else:
        print("Creating new folders.")
        # Create paths including parent directories
        path.mkdir(parents=True, exist_ok=True)


def set_logger(SAVE_OUTPUT, LOG_FILE_NAME):
    """For better handling logging functionality.

    Obtained and modified from Best practices to log from CS230 Deep Learning, Stanford.
    https://cs230-stanford.github.io/logging-hyperparams.html

    Example:
    ```
    logging.info("Starting training...")
    ```

    Attributes:
        SAVE_OUTPUT: The directory of where you want to save the logs
        LOG_FILE_NAME: The name of the log file

    Returns:
        logger: logger with the settings you specified.
    """

    logger = getLogger()
    logger.setLevel(INFO)

    if not logger.handlers:
        # Define settings for logging
        log_format = Formatter(
            '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
        # for streaming, up to INFO level
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # for file, up to DEBUG level
        handler = FileHandler(SAVE_OUTPUT + '/' + LOG_FILE_NAME, 'w')
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.setLevel(DEBUG)
        logger.addHandler(handler)

    return logger


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    return int(hours), int(minutes), seconds


def chunking(x, y, batch_size, future_step, num_chan, num_chan_kin):
    """
    A function to chunk the data into a batch size
    :param x: Input data
    :param y: Output target
    :param future_step: How much further you want to predict
    :return: Chunked matrices both for input and output
    """
    # Initialize the sequence and the next value
    seq, next_val = [], []
    # seq = np.empty(shape=(len(x)-batch_size-future_step, batch_size, future_step))
    # next_val = np.empty(shape=(len(y)+batch_size+future_step-1, num_chan_kin))
    # Based on the batch size and the future step size,
    # run a for loop to create chunks.
    # Here, it's BATCH_SIZE - 1 because we are trying to predict
    # one sample ahead. You could change this to your own way
    # e.g. want to predict 5 samples ahead, then - 5
    for i in range(0, len(x) - batch_size - future_step, future_step):
        seq.append(x[i: i + batch_size, :])
        next_val.append(y[i + batch_size + future_step - 1, :])

    # So now the data is [Samples, Batch size, One step prediction]
    seq = np.reshape(seq, [-1, batch_size, num_chan])
    next_val = np.reshape(next_val, [-1, num_chan_kin])

    X = np.array(seq)
    Y = np.array(next_val)

    return X, Y


def update_args(args, best_params):
    """Update some of the parameters after optuna optimization.

    """

    if args.decode_type in args.rnn_decoders:
        # Regardless of fix_do, init_std is optimized so need to be updated
        args.init_std = float(best_params['init_std'])
        # If layers and hidden units are not fixed, it's optimized so update
        if args.fix_do == 0:
            args.rnn_num_hidden = int(best_params['rnn_num_hidden'])
            args.rnn_num_stack_layers = int(
                best_params['rnn_num_stack_layers'])
    # Do the same for TCN
    elif args.decode_type in args.cnn_decoders:
        args.tcn_num_hidden = int(best_params['tcn_num_hidden'])
        args.tcn_num_layers = int(best_params['tcn_num_layers'])
        args.tcn_kernel_size = int(best_params['tcn_kernel_size'])

    return args
