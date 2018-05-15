from collections import deque
import glob
import numpy as np
import os


class DataReader:
    """
    Provides batching generator for reading in features and targets from numpy binary files

    Parameters
    ---------
    directory   full path to directory
    coin        coin symbol. Bitcoin -> BTC, Ethereum -> ETH
    pair        base trading pair; either BTC, USD or USDT
    batch_size  number of records to use per batch
    train_size  fractional value for number of training records; [0, 0.99]; default=0.8 (80%)
    """
    def __init__(self, directory=None, coin=None, pair='BTC', batch_size=50, train_size=0.8):
        assert(coin is not None)
        assert(type(batch_size) is int)
        assert(type(train_size) is float)
        assert(0.01 <= train_size <= 0.99)

        self.directory = self.validate_directory(directory, coin.upper(), pair)
        self.coin = f'{coin.upper()}_{pair}'
        self.train_size = train_size
        self.batch_size = batch_size
        self.training_files = self.get_file_list(train_size, is_training=True)
        self.testing_files = self.get_file_list(train_size, is_training=False)

        # To be mutated
        self.features = None
        self.targets = None

    def __repr__(self):
        return f'DataReader<COIN={self.coin}>'

    def get_file_list(self, train_size=None, is_training=True):
        """
        Pattern matches on the target directory for the given coin/trading pair.
        File format: coin_base_UTCTargetTime_[feature|target].npy

        Files are matched, imported and sorted (ASCENDING) agaisnt the UTCTargetTime

        Returns
        -------
        zip of tuples: (feature_x, target_x), ..., (feature_N, target_N)
        """
        if train_size is None:
            train_size = self.train_size

        features = sorted(glob.glob(f'{self.directory}/{self.coin}*features*npy'))  # Sort on UTC time; ASCENDING
        targets = sorted(glob.glob(f'{self.directory}/{self.coin}*target*npy'))  # Sort on UTC time; ASCENDING

        assert(len(features) == len(targets))
        num_records = len(features)
        training_split = int(num_records * train_size)

        if is_training:
            return zip(features[:training_split], targets[:training_split])

        return zip(features[training_split:], targets[training_split:])


    @staticmethod
    def load_feature_file(file_path):
        """
        Loads feature file.
        Drops 'utcTime' column
        If NAN, fill with 0s.

        Parameters
        ----------
        file_path

        Returns
        -------
        Loads numpy binary format into numpy array
        """
        return np.nan_to_num(np.delete(np.load(file_path), 0, axis=1))

    @staticmethod
    def load_target_file(file_path):
        return np.load(file_path)

    def get_batch(self, is_training=True):
        """

        Returns
        -------
        yields generator of next file batch. If stop iteration is raised, generator is
        reloaded with all filepath info. Will continue to produce batch data without raising errors
        """
        if is_training:
            # Sequentially loop over training data.
            # If StopIteration, re-initialize `file_path` generator
            file_paths = self.get_file_list()
            while True:
                try:
                    features = list()
                    targets = list()
                    for step in range(self.batch_size):
                        _feat_path, _tgt_path = file_paths.__next__()
                        features.append(self.load_feature_file(_feat_path))
                        targets.append(self.load_target_file(_tgt_path))
                    yield dict(features=np.array(features), targets=np.array(targets).reshape(-1, 1))
                except StopIteration:
                    file_paths = self.get_file_list()

        if not is_training:
            # Sequentially load all testing files
            testing_files = self.get_file_list(is_training=False)
            features = list()
            targets = list()
            for _feat_path, _tgt_path in testing_files:
                features.append(self.load_feature_file(_feat_path))
                targets.append(self.load_target_file(_tgt_path))
            yield dict(features=np.array(features), targets=np.array(targets).reshape(-1, 1))

    @staticmethod
    def validate_directory(directory, coin, pair='BTC'):
        """

        Parameters
        ----------
        directory   directory string
        coin        coin symbol; Ethereum -> ETH, Digibyte ->, ZCash -> ZEC
        pair        base trading pair; either BTC, USD, USDT

        Returns
        -------
        directory string or raises AssertionError
        """
        assert(directory is not None)
        if directory[-1] == '/':
            directory = directory[:-1]

        _directory = os.path.join(directory, f'{coin}_{pair}')
        assert(os.path.isdir(_directory))
        return _directory
