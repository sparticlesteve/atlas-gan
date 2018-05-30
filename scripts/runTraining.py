#!/usr/bin/env python

"""
runTraining.py - a script for running the GAN training.

Things still missing:
    - custom variable initialization
    - generator loss
"""

# System
from __future__ import print_function
import os
import argparse
import logging

# External
import numpy as np
import torch
from torch.autograd import Variable

# Local
from atlasgan import gan

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('runTraining.py')
    add_arg = parser.add_argument
    add_arg('--input-data', default='/data0/users/sfarrell/atlas_rpv_data/RPV10_1600_250_01.npz')
    add_arg('--output-dir')
    add_arg('--noise-dim', type=int, default=64, help='Size of the noise vector')
    add_arg('--flip-labels', type=float, default=0, help='Probability to flip labels in discriminator updates')
    add_arg('--lr', type=float, default=0.0002, help='Learning rate')
    add_arg('--beta1', type=float, default=0.5, help='Adam beta1 parameter')
    add_arg('--n-train', type=int, help='Maximum number of training samples')
    add_arg('--n-epochs', type=int, default=1)
    add_arg('--n-save', type=int, default=8,
            help='Number of example generated images to save to output-dir after every epoch.')
    add_arg('--batch-size', type=int, default=64)
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    #add_arg('--hidden-dim', type=int, default=20)
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
    data = np.load(args.input_data, mmap_mode='r')['hist']
    logging.info('Loaded data with shape: %s' % (data.shape,))

    # Instantiate the model
    generator = gan.Generator(args.noise_dim)
    discriminator = gan.Discriminator()
    loss_func = torch.nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    logging.info('Generator module: \n%s\nParameters: %i' %
                 (generator, sum(p.numel() for p in generator.parameters())))
    logging.info('Discriminator module: \n%s\nParameters: %i' %
                 (discriminator, sum(p.numel() for p in discriminator.parameters())))

    # Prepare training summary information
    dis_outputs_real = np.zeros(args.n_epochs)
    dis_outputs_fake = np.zeros(args.n_epochs)
    dis_losses = np.zeros(args.n_epochs)
    gen_losses = np.zeros(args.n_epochs)
    gen_samples = np.zeros((args.n_epochs, args.n_save, data.shape[1], data.shape[2]))

    # Finalize the training set.
    # Note I am throwing away the last partial batch.
    n_train = data.shape[0]
    if args.n_train is not None and args.n_train > 0:
        n_train = min(n_train, args.n_train)
    n_batches = n_train // args.batch_size
    n_samples = n_batches * args.batch_size
    batch_idxs = np.arange(0, n_samples, args.batch_size)
    logging.info('Training samples: %i' % n_samples)
    logging.info('Batches per epoch: %i' % n_batches)

    # Constants
    real_labels = Variable(torch.ones(args.batch_size))
    fake_labels = Variable(torch.zeros(args.batch_size))

    # Offload to GPU
    if args.cuda:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
        loss_func = loss_func.cuda()
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()

    # Loop over epochs
    for i in range(args.n_epochs):
        logging.info('Epoch %i' % i)

        # Loop over batches
        for j in batch_idxs:
            batch_data = data[j:j+args.batch_size][:, None].astype(np.float32)

            # Label flipping for discriminator training
            flip = (np.random.random_sample() < args.flip_labels)
            d_labels_real = fake_labels if flip else real_labels
            d_labels_fake = real_labels if flip else fake_labels

            # Train discriminator with real samples
            discriminator.zero_grad()
            batch_real = Variable(torch.from_numpy(batch_data))
            if args.cuda:
                batch_real = batch_real.cuda()
            d_output_real = discriminator(batch_real)
            d_loss_real = loss_func(d_output_real, d_labels_real)
            d_loss_real.backward()
            # Train discriminator with fake generated samples
            batch_noise = Variable(torch.FloatTensor(args.batch_size, args.noise_dim, 1, 1).normal_(0, 1))
            if args.cuda:
                batch_noise = batch_noise.cuda()
            batch_fake = generator(batch_noise)
            d_output_fake = discriminator(batch_fake.detach())
            d_loss_fake = loss_func(d_output_fake, d_labels_fake)
            d_loss_fake.backward()
            # Update discriminator parameters
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_optimizer.step()

            # Train generator to fool discriminator
            generator.zero_grad()
            g_output_fake = discriminator(batch_fake)
            # We use 'real' labels for generator cost
            g_loss = loss_func(g_output_fake, real_labels)
            g_loss.backward()
            # Update generator parameters
            g_optimizer.step()

            # Mean discriminator outputs
            dis_outputs_real[i] += (d_output_real.mean() / n_batches)
            dis_outputs_fake[i] += (d_output_fake.mean() / n_batches)

            # Save losses
            dis_losses[i] += (d_loss.mean() / n_batches)
            gen_losses[i] += (g_loss.mean() / n_batches)

            ## Zero the gradients
            #discriminator.zero_grad()
            #generator.zero_grad()
            ## Prepare batch for model input
            #batch_data = data[j:j+args.batch_size][:, None].astype(np.float32)
            #batch_real = Variable(torch.from_numpy(batch_data))
            #real_labels = Variable(torch.ones(args.batch_size))
            ## Generate some fake samples
            #batch_noise = Variable(torch.FloatTensor(args.batch_size, args.noise_dim, 1, 1).normal_(0, 1))
            #batch_fake = generator(batch_noise)
            #fake_labels = Variable(torch.zeros(args.batch_size))
            ## Apply the discriminator
            #output_real = discriminator(batch_real)
            #output_fake = discriminator(batch_fake.detach())
            ## Update the discriminator
            #d_loss_real = loss_func(output_real, real_labels)
            #d_loss_fake = loss_func(output_fake, fake_labels)
            #d_loss = d_loss_real + d_loss_fake
            #d_loss.backward()
            #d_optimizer.step()
            ## Compute output on fake samples again for generator update
            #output_fake = discriminator(batch_fake)
            #g_loss = loss_func(output_fake, real_labels)
            #g_loss.backward()
            #g_optimizer.step()

        logging.info('Avg discriminator real output: %.4f' % dis_outputs_real[i])
        logging.info('Avg discriminator fake output: %.4f' % dis_outputs_fake[i])
        logging.info('Avg discriminator loss: %.4f' % dis_losses[i])
        logging.info('Avg generator loss: %.4f' % gen_losses[i])

        # Save example generated data
        if args.output_dir is not None:
            make_path = lambda s: os.path.join(args.output_dir, s)
            # Select a random subset of the last batch of generated data
            rand_idx = np.random.choice(np.arange(args.batch_size),
                                        args.n_save, replace=False)
            gen_samples[i] = batch_fake.cpu().data.numpy()[rand_idx][:, 0]
            #gen_samples[i] = batch_fake[rand_idx][:, 0].cpu().data.numpy()
            #gen_samples[i] = batch_fake.data.numpy()[rand_idx][:, 0]

    logging.info('Finished training')

    # Save outputs
    if args.output_dir is not None:
        logging.info('Saving results to %s' % args.output_dir)
        make_path = lambda s: os.path.join(args.output_dir, s)
        # Save the models
        logging.info('Saving generator')
        torch.save(generator, make_path('generator.torch'))
        logging.info('Saving discriminator')
        torch.save(discriminator, make_path('discriminator.torch'))
        # Save the average outputs
        logging.info('Saving mean discriminator outputs')
        np.save(make_path('dis_outputs_real'), dis_outputs_real)
        np.save(make_path('dis_outputs_fake'), dis_outputs_fake)
        # Save the losses
        logging.info('Saving training losses')
        np.save(make_path('dis_losses'), dis_losses)
        np.save(make_path('gen_losses'), gen_losses)
        # Save the example generated images
        logging.info('Saving generated samples')
        np.save(make_path('gen_samples'), gen_samples)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
