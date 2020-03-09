import pdb
import sys
import numpy as np
from scipy import signal
sys.path.append("../uheeg")  # You need this to use the library
from uheeg.preprocess.manipulation import chunking


def preprocess_wrapper(args, dsets, sep_fraction, sc_eeg, sc_kin, data):
    """A wrapper that wraps:
    downsampling, standardization, train_test_splitting, chunking

    """

    data_key = ('eeg_train', 'eeg_test', 'kin_train', 'kin_test')
    all_data = {}
    # ===== Unwrap data =====
    for key in data_key:
        all_data[key] = data[key]

    # ===== Down-sampling =====
    if args.downsample_do:
        for key in data_key:
            all_data[key] = downsample(args, all_data[key])

    # ===== Augment kin data ====
    if args.augment_do:
        if args.augment_type == "gradient":
            vel_equivalent_train = np.gradient(all_data['kin_train'])
            val_equivalent_test = np.gradient(all_data['kin_test'])
            all_data['kin_train'] = np.hstack(
                (all_data['kin_train'], vel_equivalent_train[0]))
            all_data['kin_test'] = np.hstack(
                (all_data['kin_test'], val_equivalent_test[0]))

    # ===== Standardizing =====
    if args.standardize_do:
        _, all_data['eeg_train'], all_data['eeg_test'] = standardize_dataset(sc_eeg,
                                                                             all_data['eeg_train'],
                                                                             all_data['eeg_test'])
        sc_kin, all_data['kin_train'], all_data['kin_test'] = standardize_dataset(sc_kin,
                                                                                  all_data['kin_train'],
                                                                                  all_data['kin_test'])

    # ===== Splitting into train and validation =====
    eeg_train, eeg_valid = train_valid_separation(
        sep_fraction, all_data['eeg_train'])
    kin_train, kin_valid = train_valid_separation(
        sep_fraction, all_data['kin_train'])
    eeg, kin = {}, {}
    eeg['train'], eeg['valid'], eeg['test'] = eeg_train, eeg_valid, all_data['eeg_test']
    kin['train'], kin['valid'], kin['test'] = kin_train, kin_valid, all_data['kin_test']

    # Use the helper function to prepare the data (using sliding windows simulating real-time)
    # dimension: train - [num_samples-tap_size-future_step, tap_size, num_eeg_chan]
    #            test  - [num_samples-tap_size-future_step, num_kin_joints]
    X, X_2d, Y = {}, {}, {}
    for d in dsets:
        if args.augment_do:
            X[d], Y[d] = chunking(eeg[d], kin[d], args.tap_size,
                                  args.future_step, args.num_chan_eeg, args.num_chan_kin*2)
        else:
            X[d], Y[d] = chunking(eeg[d], kin[d], args.tap_size,
                                  args.future_step, args.num_chan_eeg, args.num_chan_kin)

        # for filters that only take 2D input
        # dimension: train_2d - [num_samples-tap_size-1, tap_size*num_eeg_chan]
        X_2d[d] = X[d].reshape(X[d].shape[0], (X[d].shape[1]*X[d].shape[2]))

    return X, X_2d, Y, sc_kin


def downsample(args, data):
    """A function to downsample.

    Using decimate instead of resample because decimate has anti-aliasing.
    """

    downsampled_data = signal.decimate(data, args.downsample_factor, axis=0)

    return downsampled_data


def standardize_dataset(sc, train_data, test_data):
    """Standardize the dataset.

    Arguments:
        sc:         standardize class from sklearn "StandardScaler()"
        train_data: the train data you want to standardize
        test_data:  the test data you want to standardize based on the train data

    Returns:
        sc:         standardize class fit to the data. Used later for transformation.
        train_stan: standardized train data
        test_stan:  standardized test data
    """

    train_stan = sc.fit_transform(train_data)
    test_stan = sc.transform(test_data)

    return sc, train_stan, test_stan


def train_valid_separation(sep_fraction, data_to_sep):
    """A function to separate the data into training and validation.

    Arguments:
        sep_fraction:   How much portion of the data you want to use for train
                        e.g. 0.8 for 80%
        data_to_sep:    The data you want to separate.

    Returns:

    """

    t_samp = int(data_to_sep.shape[0]*sep_fraction)
    t_data = data_to_sep[:t_samp, :]
    v_data = data_to_sep[t_samp+1:, :]

    return t_data, v_data
