from collections import deque
import glob
import numpy as np
import re
import os

utc_time_pattern = re.compile(r'_([0-9]+)')


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
        #assert(coin is not None)
        assert(type(batch_size) is int)
        assert(type(train_size) is float)
        assert(0.01 <= train_size <= 0.99)

        self.directory = self.validate_directory(directory, coin, pair)
        self.coin = f'{coin}_{pair}' if coin is not None else None
        self.train_size = train_size
        self.batch_size = batch_size
        self.training_file = os.path.join(self.directory, 'manifest.train.txt')
        self.testing_file = os.path.join(self.directory, 'manifest.test.txt')
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

        # Train on specific coin
        if self.coin is not None:

            features = sorted(glob.glob(f'{self.directory}/{self.coin}*features*npy'))  # Sort on UTC time; ASCENDING
            targets = sorted(glob.glob(f'{self.directory}/{self.coin}*target*npy'))  # Sort on UTC time; ASCENDING

            assert(len(features) == len(targets))
            num_records = len(features)
            training_split = int(num_records * train_size)

            if is_training:
                return zip(features[:training_split], targets[:training_split])

            return zip(features[training_split:], targets[training_split:])

        # Train on all coins
        if self.coin is None:
            features = list()
            targets = list()
            for _, _, files in os.walk(self.directory):

                if len(files) == 0:
                    continue

                file_list = sorted(files)
                subdir_features = [f for f in file_list if 'features' in f]
                subdir_targets = [t for t in file_list if 'target' in t]

                assert(len(subdir_features) == len(subdir_targets))
                num_records = len(subdir_features)
                training_split = int(num_records * train_size)

                if is_training:
                    features = features + subdir_features[:training_split]
                    targets = targets + subdir_targets[:training_split]
                else:
                    features = features + subdir_features[training_split:]
                    targets = targets + subdir_targets[training_split:]

            return zip(features, targets)

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
        #target_utc_time = re.search(utc_time_pattern, file_path).group(1)
        #return {'time': target_utc_time, 'target': np.load(file_path)}
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
                        targets.append(self.load_target_file(_tgt_path).get('target'))
                    yield dict(features=np.array(features), targets=np.array(targets).reshape(-1, 1))
                except StopIteration:
                    file_paths = self.get_file_list()

        if not is_training:
            # Sequentially load all testing files
            testing_files = self.get_file_list(is_training=False)
            features = list()
            targets = list()
            utctimes = list()
            for _feat_path, _tgt_path in testing_files:
                features.append(self.load_feature_file(_feat_path))
                _tgt_dict = self.load_target_file(_tgt_path)
                targets.append(_tgt_dict.get('target'))
                utctimes.append(_tgt_dict.get('time'))
            yield dict(features=np.array(features), targets=np.array(targets).reshape(-1, 1), times=np.array(utctimes))

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
        _directory = directory
        if _directory[-1] == '/':
            _directory = _directory[:-1]

        if coin is not None:
            coin = coin.upper()
            _directory = os.path.join(_directory, f'{coin}_{pair}')

        assert(os.path.isdir(_directory))
        return _directory

    def create_train_test_manifest(self, overwrite=False):
        """
        Walk directory tree and creates a manifest of file names to be used
        for training and test.

        The files can then be passed as input to a generator object to load
        on demand the required parameters.


        Returns
        -------
        NoneType
        """

        if overwrite:
            try:
                os.remove(self.training_file)
                os.remove(self.testing_file)
            except FileExistsError:
                pass

        # Walk the directory tree
        # Generate file list for training/test
        # Use file list for generator consumption of desired files
        for dirpath, _, files in os.walk(self.directory):

            if (len(files) == 0) or (dirpath == self.directory):
                continue

            # Ensure training split is an even number
            # The requires matching feature/target files are written to each manifest
            training_split = int(self.train_size * len(files))
            if not training_split % 2 == 0:
                training_split += 1

            # Split files into training and test
            files = sorted(files)
            training_files = files[:training_split]
            self.write_files_to_manifest(dirpath, self.training_file, training_files)

            testing_files = files[training_split:]
            self.write_files_to_manifest(dirpath, self.testing_file, testing_files)
        return

    @staticmethod
    def write_files_to_manifest(directory=None, file_name=None, file_list=None):
        assert(directory is not None)
        assert(file_name is not None)
        assert(file_list is not None)

        with open(file_name, 'a+') as fout:
            for file in file_list:
                fout.write(f'{directory}/{file }\n')

    def read_file_in_batch_size(self, train_or_test=None, batch_size=None):
        """
        Use a generator it dynamically load training/testing data
        Parameters
        ----------
        train_or_test: either 'train' or 'test'
        batch_size: integer of batch size
        """
        assert(self.coin is None)
        assert(train_or_test is not None)
        assert(train_or_test in ['train', 'test'])

        if batch_size is None:
            batch_size = self.batch_size

        file_path = self.training_file if train_or_test == 'train' else self.testing_file

        try:
            f = open(file_path)
            while True:
                features = list()
                targets = list()

                lines = [f.readline().strip() for _ in range(batch_size * 2)]  # 2x for feature and target
                [features.append(self.load_feature_file(f)) for f in lines if 'feature' in f]
                [targets.append(self.load_target_file(t)) for t in lines if 'target' in t]
                yield {'features': np.array(features), 'targets': np.array(targets).reshape(-1, 1)}
        except StopIteration:
            self.read_file_in_batch_size(train_or_test, batch_size)
