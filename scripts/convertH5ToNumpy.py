"""
This script will convert the delphes HDF5 files into numpy archives,
dropping all of the individual 'event' groups and just keeping the
'all_events' group data.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys

import h5py
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('input_file')
    parser.add_argument('output_file_prefix')
    parser.add_argument('--n-test', type=int, default=8192)
    parser.add_argument('--n-valid', type=int, default=8192)
    args = parser.parse_args()

    # Hardcoded list of keys to save (for now)
    keys = ['hist']

    # Open the HDF5 file and get the relevant data
    data = h5py.File(args.input_file)['all_events']

    # Count how many events we have in this file
    n_total = data['weight'].shape[0]
    n_train = n_total - args.n_valid - args.n_test

    # Split the data
    train_data, valid_data, test_data = {}, {}, {}
    for key in keys:
        train_data[key] = data[key][:n_train]
        valid_data[key] = data[key][n_train:n_train+args.n_valid]
        test_data[key] = data[key][n_train+args.n_valid:]

    train_file = args.output_file_prefix + '_train.npz'
    valid_file = args.output_file_prefix + '_valid.npz'
    test_file = args.output_file_prefix + '_test.npz'
    np.savez(train_file, **train_data)
    np.savez(valid_file, **valid_data)
    np.savez(test_file, **test_data)

if __name__ == '__main__':
    main()
