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
    parser.add_argument('output_file')
    args = parser.parse_args()

    # Open the HDF5 file and get the relevant data
    data = h5py.File(args.input_file)['all_events']

    #for k, x in data.items():
    #    print(k, x.shape)
    #print(list(data.keys()))
    np.savez(args.output_file, **data)

if __name__ == '__main__':
    main()
