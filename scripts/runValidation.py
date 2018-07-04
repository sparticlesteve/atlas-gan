#!/usr/bin/env python

"""
evalHPO.py - a script for evaluating HP training runs.
"""

# Compatibility
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# System
import os
import argparse
import logging
import json
import multiprocessing as mp
from functools import partial

# Externals
import numpy as np
import torch
from torch.autograd import Variable

# Locals
from atlasgan.datasets import RPVImages, inverse_transform_data, generate_noise
from atlasgan.reco import compute_physics_variables
from atlasgan.validate import compute_metrics
from atlasgan import gan


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('evalHPO.py')
    add_arg = parser.add_argument
    add_arg('--input-data', default='/global/cscratch1/sd/sfarrell/atlas_gan/data_split/RPV10_1400_850_01_valid.npz')
    add_arg('--train-dir', required=True, help='Training results directory to analyze')
    add_arg('--output-file-name', default='validation_metrics.npz')
    add_arg('--n-valid', type=int, default=4096, help='Number of validation samples')
    add_arg('--image-norm', type=float, default=4e6,
            help='Normalization factor for the image data')
    add_arg('--n-workers', type=int, default=1, help='Number of process workers')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def write_metrics(output_dir, metrics, output_file_name):
    metrics_file = os.path.join(output_dir, output_file_name)
    logging.info('Writing metrics to %s' % metrics_file)
    np.savez(metrics_file, **metrics)

def load_model_config(train_dir):
    with open(os.path.join(train_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    return config

def load_model(train_dir, checkpoint_id, model_config):
    """Load generator and discriminator from checkpoint"""
    checkpoint_file = os.path.join(
        train_dir, 'checkpoints', 'model_checkpoint_%03i.pth.tar' % checkpoint_id
    )
    # Load the checkpoint and map onto CPU
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    generator = gan.Generator(model_config['noise_dim'],
                              threshold=model_config['threshold'],
                              n_filters=model_config['n_filters'])
    discriminator = gan.Discriminator(n_filters=model_config['n_filters'])
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    # Ensure the model is in eval mode
    return generator.eval(), discriminator.eval()

def compute_epoch_metrics(epoch, train_dir, model_config,
                          valid_noise, scale, real_vars):
    logging.info('Epoch %i: loading model' % epoch)
    generator, discriminator = load_model(train_dir, epoch, model_config)
    # Generate images
    logging.info('Epoch %i: Generating images' % epoch)
    valid_fake = generator(valid_noise)
    fake_images = inverse_transform_data(valid_fake.data.numpy().squeeze(1), scale)
    # Compute reconstructed physics variables from fake images
    logging.info('Epoch %i: Computing physics variables' % epoch)
    fake_vars = compute_physics_variables(fake_images)
    # Compute validation metrics
    logging.info('Epoch %i: Computing metrics' % epoch)
    metrics = compute_metrics(real_vars, fake_vars)
    metrics['epoch'] = epoch
    return metrics

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

    # Load the model config
    config = load_model_config(args.train_dir)

    # Load the data
    # FIXME: Deterministically define both validation and test set
    dataset = RPVImages(args.input_data, n_samples=args.n_valid,
                        scale=args.image_norm, from_back=True)
    logging.info('Loaded data with shape: %s' % str(dataset.data.size()))

    # Prepare and reconstruct the real images
    logging.info('Preparing and reconstructing real validation images')
    valid_real = Variable(dataset.data, volatile=True)
    # Random noise; input for generator
    valid_noise = Variable(generate_noise(args.n_valid, config['noise_dim']), volatile=True)
    # Get the numpy format of real images
    scale = args.image_norm
    real_images = inverse_transform_data(dataset.data.numpy().squeeze(1), scale)
    # Compute reconstructed physics variables from real images
    real_vars = compute_physics_variables(real_images)

    # Load all model checkpoints
    train_summaries = np.load(os.path.join(args.train_dir, 'summaries.npz'))
    epochs = train_summaries['epoch']

    # Compute metrics for all model checkpoints
    pool = mp.Pool(processes=args.n_workers)
    func = partial(compute_epoch_metrics, train_dir=args.train_dir,
                   model_config=config, valid_noise=valid_noise,
                   real_vars=real_vars, scale=scale)
    metrics_list = pool.map(func, epochs)

    # Convert to dict of metrics
    metrics = {}
    for key in metrics_list[0].keys():
        metrics[key] = [m[key] for m in metrics_list]
    write_metrics(args.train_dir, metrics, args.output_file_name)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
