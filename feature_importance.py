import pdb
import csv
import gc
import argparse
import time
import yaml
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from tqdm import tqdm
# Your own functions
from datasets.make_dataset import make_datasets
from trainer.train_and_eval import define_model, train_wrapper, test, objective, objective_2d, objective_fix
from utils.preprocess import preprocess_wrapper
from utils.torch_preprocess import torch_dataloaders
from utils.validation import calc_rval
from utils.utils import set_logger, timer, dir_maker, update_args
from utils.ymlfun import get_args, get_parser
# for debugging, you could delete this later


if __name__ == '__main__':
    """Running feature of importance analysis

    You may need to modify some of the variables and paths to run the code.
    Configurations are taken by the config_feature_importance.yaml file and the argparse arguments
    are automatically generated from the YAML file. So you could specify the args
    from CLI as long as it's written in the config file.
    """

    # Load configuration file
    DIR_CURRENT = Path(__file__).parent
    args = get_args(get_parser(str(DIR_CURRENT / "config_feature_importance.yaml")))
    args.config = None  # No longer needed

    # ===== Define paths =====
    DATA_EEG = DIR_CURRENT / 'data' / 'raw' / 'avatar' / 'eeg' / args.data_path
    DATA_KIN = DIR_CURRENT / 'data' / 'raw' / 'avatar' / 'kin' / 'SL'
    DIR_DATA_ALL = DIR_CURRENT / 'data' / 'processed' / args.data_compressed
    DIR_DATA_EACH = DIR_CURRENT / 'data' / \
        'processed' / args.data_compressed / 'each_trial'
    DIR_LOG = DIR_CURRENT / 'results' / 'logs'
    folder_name = str(args.exp + "_" + args.decode_type +
                      "_" + str(args.tap_size))
    # TODO: Fix below checkpoint path when open sourcing the code
    DIR_CHECK = DIR_CURRENT / 'results' / 'checkpoints' / folder_name
    # DIR_CHECK = Path("/media/snakagom/UUI/Dropbox/UH/phd_aim1/results/checkpoints") / folder_name
    DIR_FIMPORTANCE = DIR_CURRENT / 'results' / 'feature_importance' / folder_name
    DIR_RESULTS_SUMMARY_EACH = DIR_CURRENT / 'results' / \
        'summary' / folder_name / 'each_trial'
    # Define log file
    log_fname = "fi_" + args.exp + "_" + args.decode_type + \
        "_" + str(args.tap_size) + args.name_log
    log = set_logger(str(DIR_LOG), log_fname)
    # Check the directories to save files and create if it doesn't exist
    dir_maker(DIR_DATA_ALL)
    dir_maker(DIR_DATA_EACH)
    dir_maker(DIR_LOG)
    dir_maker(DIR_CHECK)
    dir_maker(DIR_FIMPORTANCE)
    dir_maker(DIR_RESULTS_SUMMARY_EACH)

    CHAN_INFO = Path("chan46.csv")
    assert CHAN_INFO.is_file(), "Channel location file doesn't exist."
    with open(str(CHAN_INFO)) as csvfile:
        chan_info = list(csv.reader(csvfile))[0]
    chan_info.append('Baseline') # We will run without any permutation at last

    # use GPU if available
    args.cuda = torch.cuda.is_available()

    # Just for checking purposes
    log.info("========================================")
    log.info(
        f"Parameters: {yaml.dump(vars(args), None, default_flow_style=False)}")
    log.info(f"Decode type: {args.decode_type}")
    log.info(f"GPU available: {args.cuda}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Tap size: {args.tap_size}")
    log.info("========================================")

    # set the random seed for reproducible experiments
    # https://pytorch.org/docs/master/notes/randomness.html
    torch.manual_seed(0)
    np.random.seed(0)
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
    log.info(
        f"Data loading took: {int(hours)} Hours {int(mins)} minutes {round(seconds, 3)} seconds")

    # Define some variables to keep track of the results
    num_eeg_train = len(list(DIR_DATA_EACH.glob('*.npz')))
    key_eeg_train = [i.stem for i in DIR_DATA_EACH.glob('*.npz')]
    sep_fraction = 0.8  # If 0.8, 80% for train and 20% validation
    joints = ('hip', 'knee', 'ankle')
    dsets = ('train', 'valid', 'test')
    dtype = ('eeg_train', 'eeg_test', 'kin_train', 'kin_test')
    metrics = ('Hip R-val', 'Knee R-val', 'Ankle R-val', 'MSE', 'Hip R2', 'Knee R2', 'Ankle R2')

    # ========== 2) Training for each trial ==========
    log.info("Start training each trial")
    start = time.time()
    # For each subject-trial
    for subID in tqdm(range(num_eeg_train)):
        # Trial name
        trial_ID = key_eeg_train[subID]
        f_type_word = "Ch" if args.feature_type == 0 else ""
        save_file_name = f_type_word+"_"+trial_ID+"_feature.csv"
        save_full_path = DIR_FIMPORTANCE / save_file_name
        # Check if the file exists, if it does, skip
        if save_full_path.is_file():
            log.info("The file already exists, skipping.")
        else:
            # Just to keep track
            log.info(f"Finished processing {subID} / {num_eeg_train}")
            # Get the file name for the pre-trained model
            if args.decode_type in args.input_2d_decoders:
                file_name = trial_ID + ".sav"
            elif args.decode_type in args.input_3d_decoders:
                file_name = trial_ID + ".pth.tar"

            # Get the pre-trained model
            my_file = Path(DIR_CHECK / file_name)
            assert my_file.is_file(), "Pre-trained model does not exist. Check the path, train one if haven't."

            log.debug(f"  Loading trained model Processing: {trial_ID}")

            # ===== 2.1) Load the pre-trained model =====
            if args.decode_type in args.input_2d_decoders:
                with open(str(my_file), 'rb') as file_name:
                    model = pickle.load(file_name)
            elif args.decode_type in args.input_3d_decoders:
                trial_file_name = trial_ID + "_params.yaml"
                learned_params = yaml.load(open(str(DIR_RESULTS_SUMMARY_EACH / trial_file_name)))
                args = update_args(args, learned_params)
                param_dict = torch.load(str(my_file))
                model = define_model(args)
                model.load_state_dict(param_dict)

            # ===== 2.2) Load the data =====
            dataset_name = trial_ID + ".npz"
            data = np.load(str(DIR_DATA_EACH / dataset_name))

            # Initialize the final results
            final_results = []

            # Based on different feature of importance analysis type
            if args.feature_type == 0: # Channel feature of importance
                f_type_word = "Ch"
                # Apply permutation for each channel on a trained model
                for ch in range(args.num_chan_eeg+1):
                    # Initializing. No need to permutate eeg_train, kin_train, kin_test
                    data_permutated = {}
                    for d in dtype:
                        data_permutated[d] = data[d].copy()

                    # Without any permutation for baseline performance
                    if ch == args.num_chan_eeg:
                        log.info(f" Running final round with no permutation.")
                    else:
                        # ===== 2.2) Permutate one channel =====
                        if args.permutate_type == 0: # permutate normally
                            data_permutated[dtype[1]][:, ch] = np.random.permutation(data[dtype[1]][:, ch])
                        elif args.permutate_type == 1: # replace with zeros
                            data_permutated[dtype[1]][:, ch] = np.zeros(data[dtype[1]][:, ch].shape)

                    # TODO: Later delete the following
                    log.info(data_permutated[dtype[1]][0, :] - data[dtype[1]][0, :])

                    # ===== 2.3) Preparing the data =====
                    # Define a class for eeg and kin to standardize
                    sc_eeg, sc_kin = StandardScaler(), StandardScaler()

                    # Create datasets for training, validation, and testing
                    X, X_2d, Y, sc_kin = preprocess_wrapper(args,
                                        dsets,
                                        sep_fraction,
                                        sc_eeg,
                                        sc_kin,
                                        data_permutated)
                    del data_permutated
                    gc.collect()

                    for d in dsets:
                        # Just for logging purposes
                        log.debug(f"        Chunked {d} data: {X[d].shape}")
                        log.debug(f"        Chunked {d} data 2D: {X_2d[d].shape}")
                        log.debug(f"        Chunked {d} target: {Y[d].shape}")

                    # Define dataloaders for PyTorch
                    loaders = torch_dataloaders(args, dsets, X, Y)

                    # ===== 2.5) Test the model =====
                    log.info("   --------------------------------------")
                    log.info("   Start testing the permutated data     ")
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
                    # R-values
                    r_vals = {}
                    r2_vals = {}
                    for i, j in enumerate(joints):
                        r_vals[j] = np.corrcoef(actual[:, i], pred[:, i])[0, 1]
                        r2_vals[j] = r2_score(actual[:, i], pred[:, i])
                    mse = mean_squared_error(actual, pred)

                    # For logging purposes
                    log.info(f" Evaluating the metrics:")
                    log.info(f"     R-val Hip: {r_vals[joints[0]]:.3f}")
                    log.info(f"     MSE: {mse:.3f}")
                    log.info(f"     R2 Hip: {r2_vals[joints[0]]:.3f}")

                    # Need to add in an array
                    all_r = [r_vals[joints[i]] for i in range(3)]
                    all_r2 = [r2_vals[joints[i]] for i in range(3)]
                    final_results.append([*all_r, mse, *all_r2])

                    log.info(f"   Finished processing: {trial_ID}, Channel: {ch}")

                    # garbage collection
                    del target, prediction, X, X_2d, Y, loaders
                    torch.cuda.empty_cache()
                    gc.collect()

                # Convert the final_results into a pandas dataframe
                df = pd.DataFrame(final_results)
                df.columns = metrics
                df.index = chan_info

                # Save the results
                log.info("Saving the results")
                df.to_csv(str(DIR_FIMPORTANCE)+"/"+f_type_word+"_"+trial_ID+
                           "_feature.csv", index=False, header=False)

    # ========== 3) Finished permutating all the options ==========
    end = time.time()
    log.info("Finsihed analyzing all the subjects")
    hours, mins, seconds = timer(start, end)
    log.info(
        f"Analysis took: {int(hours)} Hours {int(mins)} minutes {seconds:.2f} seconds")
