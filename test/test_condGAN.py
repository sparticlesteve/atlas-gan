"""
Test the conditional GAN model code.
"""

# Compatibility
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# Externals
import numpy as np
import torch
from torch.autograd import Variable

# Locals
from atlasgan import cgan
from atlasgan.datasets import RPVImages, generate_noise

# Load some data
print('hello')

n_samples = 64
dataset = RPVImages(
    '/global/cscratch1/sd/sfarrell/atlas_gan/data/RPV10_1400_850_01.npz',
    n_samples=n_samples,
    scale=4e6
)

print(dataset.data.shape)

# Construct the generator and discriminator
noise_dim = 32
cond_dim = 1
g = cgan.Generator(noise_dim, cond_dim=cond_dim)
d = cgan.Discriminator(cond_dim=cond_dim)

print(g)
print(d)

# Test forward pass of the models
batch_real = Variable(dataset.data)
batch_noise = Variable(generate_noise(n_samples, noise_dim))
batch_cond = Variable(torch.FloatTensor(n_samples, cond_dim).fill_(1400))

batch_fake = g(batch_noise, batch_cond)

out_real = d(batch_real, batch_cond)
out_fake = d(batch_fake, batch_cond)
