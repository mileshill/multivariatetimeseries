import numpy as np
import pandas as pd
import re
import os

csv_pattern = re.compile(r'^.*csv$')


class DataTransforms:

    def __init__(self, file_path=None):

        self.file_path = self.validate_file_path(file_path)
        self.trading_pair = self.get_trading_pair(file_path)
        self.dir_data = '/home/miles/PycharmProjects/MultivariateTimeSeries/data'

        # To be mutated
        self.df = None
        self.df_variance = None

    def __repr__(self):
        basename = os.path.basename(self.file_path)
        rows, cols = self.df.shape if self.df is not None else [None, None]
        return f'DataTransforms<file={basename}, shape=[{rows},{cols}]>'

    @staticmethod
    def get_trading_pair(file_path):
        return os.path.basename(file_path).split('.')[0]  # "XXX_BTC.csv" => "XXX_BTC"

    @staticmethod
    def validate_file_path(file_path):
        """

        Parameters
        ----------
        file_path - path to CSV

        Returns
        -------
        file_path or raises AssertionError

        """
        assert(file_path is not None)
        assert(os.path.isfile(file_path))
        assert(re.match(csv_pattern, file_path) is not None)
        return file_path

    def import_csv(self, *args, **kwargs):
        self.df = pd.read_csv(self.file_path, *args, **kwargs)

    @staticmethod
    def variance(*args):
        """

        Parameters
        ----------
        args - array of length 2

        Returns
        -------
        float represent relative variance

        """
        a, b = args[0]
        return (b - a) / a

    def compute_rolling_variance(self):
        """
        Imports the file if not previously done.
            Set index to `time`
            Drop columns

        Compute rolling variance over all columns with `window` = 2
        Set self.df_variance to the resulting dataframe
        """
        if self.df is None:
            self.import_csv()
            self.df.set_index('time', inplace=True)
            self.df.drop(labels=['localtime', 'volumefrom'], axis=1, inplace=True)

        self.df_variance = self.df.rolling(window=2).apply(lambda window: self.variance(window))

    def get_index_slices(self, timesteps=None):
        """
        Creates the slices necessary to convert the data into a training set for tensorflow LSTM cells
        Parameters
        ----------
        timesteps - sequence length for LSTM

        Returns
        -------
        Generator of pandas IndexSlices
        """
        assert(timesteps is not None)
        assert(self.df_variance is not None)

        num_records = self.df_variance.shape[0]
        return [pd.IndexSlice[i - timesteps - 1: i, :] for i in range(timesteps + 1, num_records + 1)]

    def write_lstm_data(self, timesteps=None, dir_data=None):
        """
        Write out given file into features/target for given number of timesteps

        Parameters
        ----------
        timesteps - number of rows to use for single training instance
        input_cols - columns to use within each timestep

        Returns
        -------
        None
        """
        assert(timesteps is not None)

        if dir_data is None:
            try:
                assert(os.path.isdir(self.dir_data))
                dir_data = os.path.join(self.dir_data, self.trading_pair)
                os.mkdir(dir_data)
            except FileExistsError:
                pass

        slices = self.get_index_slices(timesteps)
        for sliced in slices:
            # Obtain timestemp slice
            df_sliced = self.df_variance.iloc[sliced]

            # Force column ordering
            column_order = ['time', 'open', 'high', 'low', 'close', 'volumeto']
            features = df_sliced.iloc[:-1].reset_index()[column_order].as_matrix()
            target = df_sliced.iloc[-1:]['high'].values[0]

            # Create file paths for saving
            target_time = np.asarray(df_sliced.index)[-1]
            feature_file_name = f'{self.trading_pair}_{target_time}_features.npy'
            target_file_name = f'{self.trading_pair}_{target_time}_target.npy'

            feature_file_path = os.path.join(dir_data, feature_file_name)
            target_file_path = os.path.join(dir_data, target_file_name)

            # Save as numpy binary
            if not os.path.isfile(feature_file_path):
                np.save(feature_file_path, features)

            if not os.path.isfile(target_file_path):
                np.save(target_file_path, target)

    def build_dataset(self, timesteps=None, dir_data=None):
        self.compute_rolling_variance()
        self.write_lstm_data(timesteps=timesteps, dir_data=dir_data)


if __name__ == '__main__':
    raise NotImplementedError

