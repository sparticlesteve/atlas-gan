"""
Trainer code for the ATLAS DCGAN.
"""

# Compatibility
from __future__ import absolute_import
from __future__ import division

# System
import os
import json
import logging

# Externals
import numpy as np
import torch
from torch.autograd import Variable

# Locals
from . import gan
from .base_trainer import BaseTrainer
from .dataset import generate_noise

class DCGANTrainer(BaseTrainer):
    """
    A trainer for the ATLAS DCGAN model.

    Implements the training logic, tracks state and metrics,
    and impelemnts logging and checkpointing.
    """

    def __init__(self, noise_dim=64, n_filters=16,
                 lr=0.0002, beta1=0.5, beta2=0.999,
                 threshold=0, flip_rate=0, image_norm=4e6,
                 cuda=False, output_dir=None, **kwargs):
        """
        Construct the trainer.
        This builds the model, optimizers, etc.
        """
        super(DCGANTrainer, self).__init__(
            output_dir=output_dir, cuda=cuda,
            config=dict(noise_dim=noise_dim, n_filters=n_filters,
                        lr=lr, beta1=beta1, beta2=beta2,
                        threshold=threshold, flip_rate=flip_rate,
                        image_norm=image_norm),
            **kwargs
        )

    def build_model(self):
        """Instantiate our model"""
        self.generator = gan.Generator(noise_dim=self.config['noise_dim'],
                                       n_filters=self.config['n_filters'],
                                       threshold=self.config['threshold'])
        self.discriminator = gan.Discriminator(n_filters=self.config['n_filters'])
        self.loss_func = torch.nn.BCELoss()
        betas = (self.config['beta1'], self.config['beta2'])
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.config['lr'], betas=betas)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.config['lr'], betas=betas)

    def train_epoch(self, data_loader, n_save):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()

        # Compute number of batches
        n_batches = len(data_loader.dataset) // data_loader.batch_size

        # Initialize training summary information
        summary = dict(d_train_loss=0, g_train_loss=0,
                       d_train_output_real=0, d_train_output_fake=0)

        # Constants
        real_labels = self.make_var(torch.ones(data_loader.batch_size))
        fake_labels = self.make_var(torch.zeros(data_loader.batch_size))

        # Loop over training batches
        for batch_data in data_loader:

            # Skip partial batches
            batch_size = batch_data.size(0)
            if batch_size != data_loader.batch_size:
                continue

            # Label flipping for discriminator training
            flip = (np.random.random_sample() < self.config['flip_rate'])
            d_labels_real = fake_labels if flip else real_labels
            d_labels_fake = real_labels if flip else fake_labels

            # Train discriminator with real samples
            self.discriminator.zero_grad()
            batch_real = self.make_var(batch_data)
            d_output_real = self.discriminator(batch_real)
            d_loss_real = self.loss_func(d_output_real, d_labels_real)
            d_loss_real.backward()
            # Train discriminator with fake generated samples
            noise_dim = self.config['noise_dim']
            batch_noise = self.make_var(generate_noise(batch_size, noise_dim))
            batch_fake = self.generator(batch_noise)
            d_output_fake = self.discriminator(batch_fake.detach())
            d_loss_fake = self.loss_func(d_output_fake, d_labels_fake)
            d_loss_fake.backward()
            # Update discriminator parameters
            d_loss = (d_loss_real + d_loss_fake) / 2
            self.d_optimizer.step()

            # Train generator to fool discriminator
            self.generator.zero_grad()
            g_output_fake = self.discriminator(batch_fake)
            # We use 'real' labels for generator cost
            g_loss = self.loss_func(g_output_fake, real_labels)
            g_loss.backward()
            # Update generator parameters
            self.g_optimizer.step()

            # Update mean discriminator output summary
            summary['d_train_output_real'] += (d_output_real.mean().data[0] / n_batches)
            summary['d_train_output_fake'] += (d_output_fake.mean().data[0] / n_batches)

            # Update loss summary
            summary['d_train_loss'] += (d_loss.mean().data[0] / n_batches)
            summary['g_train_loss'] += (g_loss.mean().data[0] / n_batches)

        # Select a random subset of the last batch of generated data
        rand_idx = np.random.choice(np.arange(data_loader.batch_size),
                                    n_save, replace=False)
        summary['gen_samples'] = batch_fake.cpu().data.numpy()[rand_idx][:, 0]

        # Print some some info for the epoch
        self.logger.info('Avg discriminator real output: %.4f' % summary['d_train_output_real'])
        self.logger.info('Avg discriminator fake output: %.4f' % summary['d_train_output_fake'])
        self.logger.info('Avg discriminator loss: %.4f' % summary['d_train_loss'])
        self.logger.info('Avg generator loss: %.4f' % summary['g_train_loss'])

        return summary
