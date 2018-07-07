"""
cgan.py - pytorch model code for conditional GAN
"""

# Externals
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator module for the conditional GAN.
    """

    def __init__(self, noise_dim, cond_dim=1,
                 output_channels=1, n_filters=16, threshold=0):
        """Construct the conditioned Generator"""
        super(Generator, self).__init__()

        # Number of filters in final generator layer
        ngf = n_filters

        # Construct the model as a sequence of layers
        self.network = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + cond_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            nn.Threshold(threshold, 0)
            # state size: (nc) x 64 x 64
        )

    def forward(self, noise, cond):
        """Computes the forward pass of the generator network"""
        # Concatenate the condition features onto the noise vector
        # FIXME: move this up into trainer and remove this class
        inputs = torch.cat([noise, cond[:, :, None, None]], dim=1)
        return self.network(inputs)

class Discriminator(nn.Module):
    """
    Discriminator module for the conditional GAN.
    """

    def __init__(self, input_channels=1, n_filters=16, cond_dim=1):
        super(Discriminator, self).__init__()
        # Number of initial filters of discriminator network
        ndf = n_filters

        # Convolutional layers
        self.network = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channels + cond_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs, cond):
        # Stack the condition onto the image as channels.
        # FIXME: move this logic up into trainer and remove this class.
        cond_channels = cond[:, :, None, None].expand(cond.size(0), cond.size(1),
                                                      inputs.size(2), inputs.size(3))
        cond_inputs = torch.cat([inputs, cond_channels], dim=1)
        return self.network(cond_inputs).squeeze()
