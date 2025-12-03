# this file is for the cnn encoder and decoder used to flatten an image into a vector

import torch.nn as nn
import numpy as np

class CNN_Image_Embedding(nn.Module):
    def __init__(self, inchannels, latentdim):
        super().__init__()

        self.encoder_net = nn.Sequential( 
            nn.Conv2d(inchannels, 8, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, latentdim) 
        )

        self.decoder_net = nn.Sequential( 
            nn.Linear(latentdim, 128 * 2 * 2),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(8, inchannels, kernel_size=4, stride=2, padding=1),
        )

    #will return the flat vector from the encoder and the reconstructed image from the vector
    def forward(self, x):
        z = self.encoder_net(x)
        x_recon = self.decoder_net(z)
        return z, x_recon
    
    def encoder(self, x):
        return self.encoder_net(x)
    
    def decoder(self, x):
        return self.decoder_net(x)