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

class DCGANTrainer():
    """
    A trainer for the ATLAS DCGAN model.

    Implements the training logic, tracks state and metrics,
    and impelemnts logging and checkpointing.
    """

    def __init__(self, noise_dim, lr, beta1, beta2=0.999,
                 cuda=False, output_dir=None):
        """
        Construct the trainer.
        This builds the model, optimizers, etc.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cuda = cuda
        self.output_dir = output_dir

        # Instantiate the model
        self.noise_dim = noise_dim
        self.generator = gan.Generator(noise_dim)
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

    def train(self, data_loader, n_epochs,
              flip_labels, n_save):
        """
        Run the model training.
        """
        # Prepare training summary information
        dis_outputs_real = np.zeros(n_epochs)
        dis_outputs_fake = np.zeros(n_epochs)
        dis_losses = np.zeros(n_epochs)
        gen_losses = np.zeros(n_epochs)
        gen_samples = np.zeros((n_epochs, n_save,
                                data_loader.dataset[0].shape[1],
                                data_loader.dataset[0].shape[2]))

        # Finalize the training set
        n_train = len(data_loader.dataset)
        n_batches = n_train // data_loader.batch_size
        n_samples = n_batches * data_loader.batch_size
        self.logger.info('Training samples: %i' % n_samples)
        self.logger.info('Batches per epoch: %i' % n_batches)

        # Constants
        real_labels = self.make_var(torch.ones(data_loader.batch_size))
        fake_labels = self.make_var(torch.zeros(data_loader.batch_size))

        # Offload to GPU
        self.discriminator = self.to_device(self.discriminator)
        self.generator = self.to_device(self.generator)

        # Loop over epochs
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)

            # Loop over batches
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
                batch_noise = self.make_var(
                    torch.FloatTensor(batch_size, self.noise_dim, 1, 1)
                    .normal_(0, 1))
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

                # Mean discriminator outputs
                dis_outputs_real[i] += (d_output_real.mean() / n_batches)
                dis_outputs_fake[i] += (d_output_fake.mean() / n_batches)

                # Save losses
                dis_losses[i] += (d_loss.mean() / n_batches)
                gen_losses[i] += (g_loss.mean() / n_batches)

            self.logger.info('Avg discriminator real output: %.4f' % dis_outputs_real[i])
            self.logger.info('Avg discriminator fake output: %.4f' % dis_outputs_fake[i])
            self.logger.info('Avg discriminator loss: %.4f' % dis_losses[i])
            self.logger.info('Avg generator loss: %.4f' % gen_losses[i])

            # Save example generated data
            if self.output_dir is not None:
                make_path = lambda s: os.path.join(self.output_dir, s)
                # Select a random subset of the last batch of generated data
                rand_idx = np.random.choice(np.arange(data_loader.batch_size),
                                            n_save, replace=False)
                gen_samples[i] = batch_fake.cpu().data.numpy()[rand_idx][:, 0]

        self.logger.info('Finished training')

        # Save outputs
        if self.output_dir is not None:
            self.logger.info('Saving results to %s' % self.output_dir)
            make_path = lambda s: os.path.join(self.output_dir, s)
            # Save the models
            self.logger.info('Saving generator')
            torch.save(self.generator, make_path('generator.torch'))
            self.logger.info('Saving discriminator')
            torch.save(self.discriminator, make_path('discriminator.torch'))
            # Save the average outputs
            self.logger.info('Saving mean discriminator outputs')
            np.save(make_path('dis_outputs_real'), dis_outputs_real)
            np.save(make_path('dis_outputs_fake'), dis_outputs_fake)
            # Save the losses
            self.logger.info('Saving training losses')
            np.save(make_path('dis_losses'), dis_losses)
            np.save(make_path('gen_losses'), gen_losses)
            # Save the example generated images
            self.logger.info('Saving generated samples')
            np.save(make_path('gen_samples'), gen_samples)
