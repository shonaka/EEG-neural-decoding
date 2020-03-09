import pdb
import gc
import sys
import argparse
import time
import yaml
import pandas as pd
import numpy as np
import torch
import optuna
import pickle
import random
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
# Your own functions
from datasets.make_dataset import make_datasets
from trainer.train_and_eval import define_model, train_wrapper, test, objective, objective_2d, objective_fix
from utils.preprocess import preprocess_wrapper
from utils.torch_preprocess import torch_dataloaders
from utils.validation import validate_results
from utils.utils import set_logger, timer, dir_maker, update_args
from utils.ymlfun import get_args, get_parser
# for debugging, you could delete this later


if __name__ == '__main__':
    """Main file for running all the experiments for decoding.

    You may need to modify some of the variables and paths to run the code.
    Configurations are taken by the config.yaml file and the argparse arguments
    are automatically generated from the YAML file. So you could specify the args
    from CLI as long as it's written in the config file.
    """

    # Load configuration file
    DIR_CURRENT = Path(__file__).parent
    args = get_args(get_parser(str(DIR_CURRENT / "config.yaml")))
    args.config = None  # No longer needed

    # ===== Define paths =====
    # - DATA_EEG:       The directory EEG .csv files as an input
    # - DATA_KIN:       The directory Kinematics .csv files as a target
    # - DIR_DATA_ALL:   The directory to save all the above processed data in .npz compressed format
    # - DIR_DATA_EACH:  The directory to save .npz compressed processed data for each trial
    # - DIR_LOG:        Where you want to save the .log files
    # - DIR_CHECK:      Where you want to save the trained model parameters
    # - DIR_PRED_CSV:   Where you want to save the predicted kinematics.csv
    # - DIR_RESULTS_SUMMARY: Where you want to save the overall summary
    # - DIR_RESULTS_SUMMARY_EACH: Where you want to save the parameters used after optimization.
    # - DIR_FIGURES:    Where you want to save the figures
    DATA_EEG = DIR_CURRENT / 'data' / 'raw' / 'avatar' / 'eeg' / args.data_path
    DATA_KIN = DIR_CURRENT / 'data' / 'raw' / 'avatar' / 'kin' / 'SL'
    DIR_DATA_ALL = DIR_CURRENT / 'data' / 'processed' / args.data_compressed
    DIR_DATA_EACH = DIR_CURRENT / 'data' / \
        'processed' / args.data_compressed / 'each_trial'
    DIR_LOG = DIR_CURRENT / 'results' / 'logs'
    folder_name = str(args.exp + "_" + args.decode_type +
                      "_" + str(args.tap_size))
    DIR_CHECK = DIR_CURRENT / 'results' / 'checkpoints' / folder_name
    DIR_PRED_CSV = DIR_CURRENT / 'results' / 'predictions' / folder_name
    DIR_RESULTS_SUMMARY = DIR_CURRENT / 'results' / 'summary' / folder_name
    DIR_RESULTS_SUMMARY_EACH = DIR_CURRENT / 'results' / \
        'summary' / folder_name / 'each_trial'
    DIR_FIGURES = DIR_CURRENT / 'results' / 'figures' / 'loss' / folder_name
    # Define log file
    log_fname = args.exp + "_" + args.decode_type + \
        "_" + str(args.tap_size) + args.name_log
    log = set_logger(str(DIR_LOG), log_fname)
    # Check the directories to save files and create if it doesn't exist
    dir_maker(DIR_DATA_ALL)
    dir_maker(DIR_DATA_EACH)
    dir_maker(DIR_LOG)
    dir_maker(DIR_CHECK)
    dir_maker(DIR_PRED_CSV)
    dir_maker(DIR_RESULTS_SUMMARY)
    dir_maker(DIR_RESULTS_SUMMARY_EACH)
    dir_maker(DIR_FIGURES)

    # Something for smoke test
    if args.smoke_test == 1:
        args.num_epochs = 1
        args.tap_size = 1
        args.rnn_num_hidden = 4

    # use GPU if available
    args.cuda = torch.cuda.is_available()

    # Just for checking purposes
    log.info("========================================")
    log.info(f"Parameters: {yaml.dump(vars(args), None, default_flow_style=False)}")
    log.info(f"Smoke test: {args.smoke_test}")
    log.info(f"Decode type: {args.decode_type}")
    log.info(f"GPU available: {args.cuda}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Tap size: {args.tap_size}")
    log.info(f"Learning rate: {args.optim_lr}")
    log.info(f"Number of epochs: {args.num_epochs}")
    log.info("========================================")

    # set the random seed for reproducible experiments
    # https://pytorch.org/docs/master/notes/randomness.html
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if args.cuda:
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ========== 1) Import and sort data ==========
    log.info("Loading data, create datasets")
    start = time.time()

    # TODO: Make this part into a module "make_dataset.py"
    dataset_name = "avatar8subs.npz"
    files_number = 24  # There should be 24 files (8 subjects x 3 trials)
    make_datasets(args, DATA_EEG, DATA_KIN, DIR_DATA_ALL,
                  DIR_DATA_EACH, dataset_name, files_number)

    log.info("Finished loading and creating datasets")
    end = time.time()
    hours, mins, seconds = timer(start, end)
    log.info(f"Data loading: {int(hours)} Hours {int(mins)} minutes {round(seconds, 3)} seconds")

    # Define some variables to keep track of the results
    num_eeg_train = len(list(DIR_DATA_EACH.glob('*.npz')))
    key_eeg_train = [i.stem for i in DIR_DATA_EACH.glob('*.npz')]
    sep_fraction = 0.8  # If 0.8, 80% for train and 20% validation
    seg_len = 200
    joints = ('hip', 'knee', 'ankle')
    dsets = ('train', 'valid', 'test')

    # Initialize metrics for results logging
    mse_all = {}
    r2_all = {}
    median_rvals_all = {}

    # ========== 2) Training for each trial ==========
    log.info("Start training each trial")
    start = time.time()
    for subID in tqdm(range(num_eeg_train)):
        log.info(f"Finished processing {subID} / {num_eeg_train}")

        # Trial name
        trial_ID = key_eeg_train[subID]

        # Check if the file exists
        if args.decode_type in args.input_2d_decoders:
            file_name = trial_ID + ".sav"
        elif args.decode_type in args.input_3d_decoders:
            file_name = trial_ID + ".pth.tar"
        # So that you won't run the training again
        my_file = Path(DIR_CHECK / file_name)
        if my_file.is_file():
            log.info("The file already exists, skipping.")
        else:
            log.debug(f"  Processing: {trial_ID}")

            # ===== 2.1) Load the data =====
            dataset_name = trial_ID + ".npz"
            data = np.load(str(DIR_DATA_EACH / dataset_name))

            # ===== 2.2) Preparing the data =====
            # Define a class for eeg and kin to standardize
            sc_eeg, sc_kin = StandardScaler(), StandardScaler()

            # Create datasets for training, validation, and testing
            X, X_2d, Y, sc_kin = preprocess_wrapper(args,
                                                    dsets,
                                                    sep_fraction,
                                                    sc_eeg,
                                                    sc_kin,
                                                    data)
            del data
            gc.collect()

            for d in dsets:
                # Just for logging purposes
                log.debug(f"        Chunked {d} data: {X[d].shape}")
                log.debug(f"        Chunked {d} data 2D: {X_2d[d].shape}")
                log.debug(f"        Chunked {d} target: {Y[d].shape}")

            # Define dataloaders for PyTorch
            loaders = torch_dataloaders(args, dsets, X, Y)

            # ===== Optional: Hyperparameter tuning =====
            # If you are doing hyperparameter optimization using optuna
            if args.optuna_do:
                study_name = args.exp + "_" + trial_ID
                # For non-neural networks
                if args.decode_type in args.input_2d_decoders:
                    study = optuna.create_study(study_name=study_name,
                                                pruner=optuna.pruners.SuccessiveHalvingPruner())
                    study.optimize(lambda trial: objective_2d(trial, args, X_2d, Y, DIR_CHECK),
                                   n_trials=args.optuna_trials)
                # For neural networks
                elif args.decode_type in args.input_3d_decoders:
                    study = optuna.create_study(
                        study_name=study_name, pruner=optuna.pruners.SuccessiveHalvingPruner())
                    if args.fix_do:
                        study.optimize(lambda trial: objective_fix(trial, args, loaders, DIR_CHECK),
                                       n_trials=args.optuna_trials)
                    else:
                        study.optimize(lambda trial: objective(trial, args, loaders, DIR_CHECK),
                                       n_trials=args.optuna_trials)
                # Extract the best optimized parameters and log them
                best_params = study.best_params
                best_error = study.best_value
                log.info(f"Best parameters are: {best_params}")
                log.info(f"Best error_rate is: {best_error:.4f}")
                if args.decode_type in args.input_2d_decoders:
                    # Load the best model
                    full_path = str(
                        Path(DIR_CHECK / str(study.best_trial.trial_id)))+".sav"
                    with open(full_path, 'rb') as file_name:
                        model = pickle.load(file_name)
                elif args.decode_type in args.input_3d_decoders:
                    # Load the parameters from the best trial
                    args = update_args(args, best_params)
                    model = define_model(args)
                    full_path = str(
                        Path(DIR_CHECK / str(study.best_trial.trial_id)))+".pth.tar"
                    model.load_state_dict(torch.load(full_path))
                del study
                gc.collect()
            else:  # Not using optuna, just define the model
                # ===== 2.3) Define the model =====
                model = define_model(args)

                # ===== 2.4) Train and Validate the model =====
                log.info("   --------------------------------------")
                log.info("   Start training")
                log.info("   --------------------------------------")

                # For models that takes 2D input (ML)
                if args.decode_type in args.input_2d_decoders:
                    # This is the same for all the above models
                    model.train(X_2d['train'], Y['train'])
                    # Now start to validate
                    log.info("   Run validation")
                    if (args.decode_type == "KF") or (args.decode_type == "UKF"):
                        # prediction = model.process(X_2d['valid'], Y['valid'][0, :].T)
                        print("Skip for now")
                    else:
                        # prediction = model.predict(X_2d['valid'])
                        print("Skip for now")
                # For models that takes 3D input (DL)
                elif args.decode_type in args.input_3d_decoders:
                    # Calling your own function for training and validating
                    train_wrapper(args, model, DIR_FIGURES, trial_ID, loaders)

                log.info("   --------------------------------------")
                log.info("   Finished training")
                log.info("   --------------------------------------")

            # ===== 2.5) Test the model =====
            log.info("   --------------------------------------")
            log.info("   Start testing")
            log.info("   --------------------------------------")

            if args.decode_type in args.input_2d_decoders:
                if (args.decode_type == "KF") or (args.decode_type == "UKF"):
                    prediction = model.process(X_2d['test'], Y['test'][0, :].T)
                else:
                    prediction = model.predict(X_2d['test'])
            elif args.decode_type in args.input_3d_decoders:
                if args.decode_type in args.rnn_decoders:
                    prediction = test(model, X['test'], rnn_flag=True)
                elif args.decode_type in args.cnn_decoders:
                    prediction = test(model, X['test'])

            log.info("   --------------------------------------")
            log.info("   Finished testing")
            log.info("   --------------------------------------")

            # ===== 2.6) Post processing the results =====
            target = Y['test']
            # Scale both the prediciton using the same scaler used during the training
            if args.standardize_do:
                prediction = sc_kin.transform(prediction)
                target = sc_kin.transform(target)

            # extract right hip, knee, ankle
            actual = target[:, 0:3]
            act_size = np.array(actual).shape
            pred = prediction[:act_size[0], 0:3]

            # ===== 2.7) Validate the results using various metrics =====
            results = validate_results(joints, seg_len, actual, pred)

            # For logging purposes
            log.info(f" Testing metrics:")
            log.info(f"     MSE: {results['mse']:.3f}")
            log.info(f"     R2: {results['r2']:.3f}")
            mse_all[trial_ID] = round(np.asscalar(results['mse']), 4)
            r2_all[trial_ID] = round(np.asscalar(results['r2']), 4)
            median_rvals = {}
            for j in joints:
                log.info(f"     median r-value for {j}: {results['median_rval'][j]:.3f}")
                median_rvals[j] = round(
                    np.asscalar(results['median_rval'][j]), 4)
            median_rvals_all[trial_ID] = median_rvals

            # ===== 2.8) Save the data and model =====
            df_actual = pd.DataFrame(target)
            df_pred = pd.DataFrame(prediction)
            # Now acutally save it in a csv format
            df_pred.to_csv(str(DIR_PRED_CSV)+"/"+trial_ID +
                           "_pred.csv", index=False, header=False)
            df_actual.to_csv(str(DIR_PRED_CSV)+"/"+trial_ID +
                             "_actual.csv", index=False, header=False)

            # Save the model
            if args.decode_type in args.input_2d_decoders:
                save_model = str(Path(DIR_CHECK / trial_ID))+".sav"
                with open(save_model, 'wb') as file_name:
                    pickle.dump(model, file_name)
            elif args.decode_type in args.input_3d_decoders:
                save_path_model = str(Path(DIR_CHECK / trial_ID))+".pth.tar"
                torch.save(model.state_dict(), save_path_model)

            log.info("   Finished processing: {}".format(trial_ID))

            file_name = trial_ID + "_params.yaml"
            dict_params = vars(args)
            final_params = {}
            # Replace with best params
            for k1, v1 in dict_params.items():
                if args.optuna_do:
                    for k2, v2 in best_params.items():
                        if k1 == k2:
                            final_params[k1] = str(v2)
                        else:
                            final_params[k1] = str(v1)
                else:
                    final_params[k1] = v1
            with open(str(DIR_RESULTS_SUMMARY_EACH / file_name), 'w') as output_f:
                yaml.dump(final_params, output_f, default_flow_style=False)

            # garbage collection
            del target, prediction, df_pred, df_actual, X, X_2d, Y, model, loaders, results
            torch.cuda.empty_cache()
            gc.collect()

    # ========== 3) Finished training for all the trials ==========
    end = time.time()
    log.info("Finsihed training all the subjects")
    hours, mins, seconds = timer(start, end)
    log.info(
        f"Training and testing: {int(hours)} Hours {int(mins)} minutes {seconds:.2f} seconds")

    # Save the summary results
    dict_params = {}
    dict_params['mse'] = mse_all
    dict_params['r2'] = r2_all
    dict_params['med_rval'] = median_rvals_all
    file_name = args.decode_type + "_summary.yaml"
    with open(str(DIR_RESULTS_SUMMARY / file_name), 'w') as output_file:
        yaml.dump(dict_params, output_file, default_flow_style=False)
