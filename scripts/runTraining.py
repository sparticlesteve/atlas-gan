#!/usr/bin/env python

"""
runTraining.py - a script for running the GAN training.

Things still missing:
    - custom variable initialization
    - generator loss
"""

# System
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import argparse
import logging

# External
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Local
from atlasgan.dataset import RPVImages
from atlasgan.trainers import DCGANTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('runTraining.py')
    add_arg = parser.add_argument
    add_arg('--input-data', default='/data0/users/sfarrell/atlas_rpv_data/RPV10_1600_250_01.npz')
    add_arg('--output-dir')
    add_arg('--noise-dim', type=int, default=64, help='Size of the noise vector')
    add_arg('--n-filters', type=int, default=16,
            help='Number of initial filters in discriminator')
    add_arg('--flip-labels', type=float, default=0,
            help='Probability to flip labels in discriminator updates')
    add_arg('--lr', type=float, default=0.0002, help='Learning rate')
    add_arg('--beta1', type=float, default=0.5, help='Adam beta1 parameter')
    add_arg('--n-train', type=int, help='Maximum number of training samples')
    add_arg('--n-epochs', type=int, default=1)
    add_arg('--n-save', type=int, default=8,
            help='Number of example generated images to save to output-dir after every epoch.')
    add_arg('--image-norm', type=float, default=4e6, #6072947
            help='Normalization factor for the image data')
    add_arg('--batch-size', type=int, default=64)
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--cuda', action='store_true')
    return parser.parse_args()

def main():
    """Main program function."""
    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Load the data
    dataset = RPVImages(args.input_data, n_samples=args.n_train, scale=args.image_norm)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    logging.info('Loaded data with shape: %s' % str(dataset.data.size()))

    # Instantiate the trainer
    trainer = DCGANTrainer(noise_dim=args.noise_dim, n_filters=args.n_filters,
                           lr=args.lr, beta1=args.beta1, flip_rate=args.flip_labels,
                           threshold=500./args.image_norm, image_norm=args.image_norm,
                           output_dir=args.output_dir, cuda=args.cuda)

    # Run the training
    trainer.train(data_loader, n_epochs=args.n_epochs, n_save=args.n_save)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
