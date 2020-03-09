import pandas as pd


class csvfile():
    """
    A class for csvfile loading for deep learning projects
    """

    def __init__(self, path):
        """
        A constructor for csvfile class
        :param path: Path object from pathlib
        """
        # files = os.listdir(str(path))
        files = [x for x in path.glob('*') if x.is_file()]
        # check to make sure there's files
        assert len(files) > 0, "There's no files in the specified directory."

        # self.path = path
        self.files = files
        self.train_data = []
        self.test_data = []

    def eeg(self, num_chans, fname_len):
        """
        A function to load eeg data from csv
        :param num_chans: number of channels to kick the files with less channels
        :param fname_len: how many characters you want to retain for the key
                            e.g. 'SL04-T03-eeg-clda.csv' you want to retain 'SL04-T03' then 8.
        """
        # Initialize
        eeg_train = {}
        eeg_test = {}
        for file in self.files:
            # if the word 'clda' is in the filename
            if 'clda' in str(file):
                df = pd.read_csv(file)
                # skip the data with less columns (number of channels)
                if df.shape[1] == num_chans:
                    eeg_train[file.name[0:fname_len]] = df.values
            # if the word 'bci' is in the filename
            elif 'bci' in str(file):
                df = pd.read_csv(file)
                # skip the data with less columns
                if df.shape[1] == num_chans:
                    eeg_test[file.name[0:fname_len]] = df.values

        # make sure you put the output in the output
        self.train_data = eeg_train
        self.test_data = eeg_test

    def kin(self, fname_len):
        '''A function to load kin data from csv
        :param num_chans: number of channels to kick the files with less channels
        '''

        # Initialize
        kin_train = {}
        kin_test = {}
        for file in self.files:
            # It has to have the word 'gonio'
            if 'gonio' in str(file):
                if 'clda' in str(file):
                    df = pd.read_csv(file)
                    kin_train[file.name[0:fname_len]] = df.values
                elif 'bci' in str(file):
                    df = pd.read_csv(file)
                    kin_test[file.name[0:fname_len]] = df.values

        self.train_data = kin_train
        self.test_data = kin_test


# Other functions
def read_output_csv(files):
    kin_actual = {}
    kin_pred = {}
    for file in files:
        # Extracting trial names
        file_name = file.split('/')[-1][0:8]
        # For actual
        if 'actual' in file:
            df = pd.read_csv(file)
            kin_actual[file_name] = df.values
        # For predicted
        elif 'pred' in file:
            df = pd.read_csv(file)
            kin_pred[file_name] = df.values

    return kin_actual, kin_pred


def read_all_results(algorithms, BASE_PATH):
    '''Read prediction and ground truth from csv and put them into a dictionary

    '''

    results = {}
    for i in algorithms:
        results[i] = {}
        # Define paths
        results_path = BASE_PATH / i
        results_grab = results_path.glob('*.csv')
        # Grab all the .csv files in the results folder
        files = [str(x) for x in results_grab if x.is_file()]
        files.sort()
        print(f"{i}: {len(files)}")
        # Load the results
        actual, pred = read_output_csv(files)
        # Log them
        results[i]['actual'] = actual
        results[i]['pred'] = pred

    return results
