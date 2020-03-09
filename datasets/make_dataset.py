import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.io import csvfile


def make_datasets(args, eeg_path, kin_path, save_path_all, save_path_each, save_name, files_number):
    """A wrapper function to perform both datasets creation.

    Summary

    """

    make_dataset_all_trials(args, eeg_path, kin_path, save_path_all, save_name)
    make_dataset_per_trial(args, save_path_all,
                           save_path_each, save_name, files_number)


def make_dataset_all_trials(args, eeg_path, kin_path, save_path, save_name):
    """A function to create numpy compressed dataset.

    Summary

    Arguments:
        args:       arguments from argparse.
        eeg_path:   the directory where EEG .csv files are
        kin_path:   the directory where KIN .csv files are
        save_path:  the directory where you want to save the numpy compressed file
        save_name:  the name of the compressed file you want to save as
    """

    # First check if there's already numpy compressed files exists
    full_path = Path(save_path / save_name)
    if full_path.is_file():
        f"The compressed file with the name you specified already exists."
        return
    else:
        f"Creating the dataset."
        f"Initial run may take some time as it needs to be compressed."

        # ===== Load EEG data =====
        eeg_train, eeg_test = {}, {}
        csv_obj = csvfile(eeg_path)
        # extract first 8 characters for file recognition
        csv_obj.eeg(args.num_chan_eeg, 8)
        eeg_train, eeg_test = csv_obj.train_data, csv_obj.test_data

        # ===== Load Kinematics data =====
        csv_obj = csvfile(kin_path)
        csv_obj.kin(8)
        kin_train, kin_test = csv_obj.train_data, csv_obj.test_data

        # ===== Save the numpy compressed dataset =====
        np.savez_compressed(str(full_path),
                            eeg_train=eeg_train,
                            eeg_test=eeg_test,
                            kin_train=kin_train,
                            kin_test=kin_test)


def make_dataset_per_trial(args, save_path_all, save_path_each, save_name, files_number):
    """A function to create numpy compressed dataset per trial

    Note that it requires a whole dataset that contains all the trials.

    Arguments:

    """

    # First check if the files exists
    num_eeg_train = len(list(save_path_each.glob('*.npz')))
    if num_eeg_train == files_number:
        return

    # Load the compressed .npz file
    with np.load(str(save_path_all / save_name)) as data:
        eeg_train_all = data['eeg_train']
        eeg_test_all = data['eeg_test']
        kin_train_all = data['kin_train']
        kin_test_all = data['kin_test']

    # Get the necessarily information for processing per trial
    num_eeg_train = len(eeg_train_all.item())
    key_eeg_train = list(eeg_train_all.item().keys())

    # Processing per trial
    for subID in tqdm(range(num_eeg_train)):
        # Trial name
        trial_ID = key_eeg_train[subID]

        # First check if the compressed file already exists
        save_name = trial_ID + ".npz"
        full_path = Path(save_path_each / save_name)
        if full_path.is_file():
            f'The compressed file for {save_name} already exists. Skipping.'
            pass
        else:
            f"Creating the dataset for {save_name}."

            # Using the EEG to train, Kin to test
            # dimension: train - [num_samples, num_eeg_chan]
            #            test  - [num_samples, num_kin_joints]
            eeg_train, eeg_test = eeg_train_all.item(
            )[trial_ID], eeg_test_all.item()[trial_ID]
            kin_train, kin_test = kin_train_all.item(
            )[trial_ID], kin_test_all.item()[trial_ID]

            # Now save the dataset as .npz
            np.savez_compressed(str(full_path),
                                eeg_train=eeg_train,
                                eeg_test=eeg_test,
                                kin_train=kin_train,
                                kin_test=kin_test)
