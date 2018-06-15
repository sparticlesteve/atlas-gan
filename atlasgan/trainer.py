"""
Trainer code for the ATLAS DCGAN.
"""

# System
import os
import logging

# Externals
import numpy as np
import torch
from torch.autograd import Variable

# Locals
import gan
from dataset import generate_noise

class DCGANTrainer():
    """
    A trainer for the ATLAS DCGAN model.

    Implements the training logic, tracks state and metrics,
    and impelemnts logging and checkpointing.
    """

    def __init__(self, noise_dim, lr, beta1, beta2=0.999,
                 threshold=0, cuda=False, output_dir=None):
        """
        Construct the trainer.
        This builds the model, optimizers, etc.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cuda = cuda
        self.output_dir = output_dir
        self.summaries = {}

        # Instantiate the model
        self.noise_dim = noise_dim
        self.generator = gan.Generator(noise_dim, threshold=threshold)
        self.discriminator = gan.Discriminator()
        self.loss_func = torch.nn.BCELoss()
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr,
                                            betas=(beta1, beta2))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr,
                                            betas=(beta1, beta2))
        self.logger.info(
            'Generator module: \n%s\nParameters: %i' %
            (self.generator, sum(p.numel() for p in self.generator.parameters()))
        )
        self.logger.info(
            'Discriminator module: \n%s\nParameters: %i' %
            (self.discriminator, sum(p.numel() for p in self.discriminator.parameters()))
        )

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

    def train_epoch(self, data_loader, flip_labels, n_save):
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
            flip = (np.random.random_sample() < flip_labels)
            d_labels_real = fake_labels if flip else real_labels
            d_labels_fake = real_labels if flip else fake_labels

            # Train discriminator with real samples
            self.discriminator.zero_grad()
            batch_real = self.make_var(batch_data)
            d_output_real = self.discriminator(batch_real)
            d_loss_real = self.loss_func(d_output_real, d_labels_real)
            d_loss_real.backward()
            # Train discriminator with fake generated samples
            batch_noise = self.make_var(generate_noise(batch_size, self.noise_dim))
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

        return summary

    def train(self, data_loader, n_epochs, flip_labels, n_save):
        """Run the model training"""
        # Offload to GPU
        self.discriminator = self.to_device(self.discriminator)
        self.generator = self.to_device(self.generator)

        # Loop over epochs
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)

            # Prepare summary information
            summary = dict(epoch=i)

            # Train on this epoch
            summary.update(self.train_epoch(data_loader, flip_labels, n_save))
            self.logger.info('Avg discriminator real output: %.4f' % summary['d_train_output_real'])
            self.logger.info('Avg discriminator fake output: %.4f' % summary['d_train_output_fake'])
            self.logger.info('Avg discriminator loss: %.4f' % summary['d_train_loss'])
            self.logger.info('Avg generator loss: %.4f' % summary['g_train_loss'])

            # Save summary information
            self.save_summary(summary)

            # Model checkpointing
            self.write_checkpoint(checkpoint_id=i,
                                  generator=self.generator,
                                  discriminator=self.discriminator)

        self.logger.info('Finished training')

        # Save the combined summary information
        if self.output_dir is not None:
            self.logger.info('Saving summaries to %s' % self.output_dir)
            np.savez(os.path.join(self.output_dir, 'summaries.npz'), **self.summaries)
