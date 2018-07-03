"""
Common trainer code for the ATLAS GAN models.
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

class BaseTrainer(object):
    """
    Base class for our HEP GAN trainers.

    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, cuda=False, config={}):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.cuda = cuda
        self.output_dir = output_dir
        self.summaries = {}

        # Build the model
        self.build_model()

        # Print model summaries
        self.logger.info(
            'Generator module: \n%s\nParameters: %i' %
            (self.generator, sum(p.numel()
             for p in self.generator.parameters()))
        )
        self.logger.info(
            'Discriminator module: \n%s\nParameters: %i' %
            (self.discriminator, sum(p.numel()
             for p in self.discriminator.parameters()))
        )

        # Write the configuration right away
        if self.output_dir is not None:
            self.write_config()

    def to_device(self, x):
        """Copy object to device"""
        return x.cuda() if self.cuda else x

    def make_var(self, x):
        """Wrap in pytorch Variable and move to device"""
        return self.to_device(Variable(x))

    def save_summary(self, summaries):
        """Save summary information"""
        for (key, val) in summaries.items():
            summary_vals = self.summaries.get(key, [])
            self.summaries[key] = summary_vals + [val]

    def write_summaries(self):
        assert self.output_dir is not None
        summary_file = os.path.join(self.output_dir, 'summaries.npz')
        self.logger.info('Saving summaries to %s' % summary_file)
        np.savez(summary_file, **self.summaries)

    def write_config(self):
        """Write the trainer config to the output directory"""
        assert self.output_dir is not None
        config_file = os.path.join(self.output_dir, 'config.json')
        self.logger.info('Saving config to %s' % config_file)
        with open(config_file, 'w') as f:
            json.dump(self.config, f)

    # TODO: write resume_checkpoint method (when actually needed)
    def write_checkpoint(self, checkpoint_id, generator, discriminator):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(dict(generator=generator.state_dict(),
                        discriminator=discriminator.state_dict()),
                   os.path.join(checkpoint_dir, checkpoint_file))

    def build_model(self):
        """Virtual method to construct the model"""
        raise NotImplementedError

    def train(self, data_loader, n_epochs, n_save):
        """Run the model training"""
        # Offload to GPU if configured
        self.discriminator = self.to_device(self.discriminator)
        self.generator = self.to_device(self.generator)

        # Loop over epochs
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)
            # Prepare summary information
            summary = dict(epoch=i)
            # Train on this epoch
            summary.update(self.train_epoch(data_loader, n_save))
            # Save summary information
            self.save_summary(summary)
            # Model checkpointing
            self.write_checkpoint(checkpoint_id=i,
                                  generator=self.generator,
                                  discriminator=self.discriminator)

        self.logger.info('Finished training')

        # Save the combined summary information
        if self.output_dir is not None:
            self.write_summaries()
