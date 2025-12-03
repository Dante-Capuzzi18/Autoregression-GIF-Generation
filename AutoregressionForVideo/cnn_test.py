# this file is used to test the encoder/decoder on the first image in one of the gifs

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import time

from image_encoders import CNN_Image_Embedding

#loading encoder/decoder model
models = CNN_Image_Embedding(inchannels=1, latentdim=512)
models.load_state_dict(torch.load("cnn_autoencoder.pth"))
models.eval()

#getting the image sequence in greyscale as tensors for the encoder/AR/decoders
frames_folder = os.path.join(os.getcwd(), "image_sequences", "25.gif")  # adjust as needed
frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(".png")])
tensor_list = []
to_tensor = transforms.ToTensor()
for f in frame_files:
    frame_path = os.path.join(frames_folder, f)
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    img_tensor = to_tensor(frame).unsqueeze(0)
    tensor_list.append(img_tensor)

#Testing encoder/decoder input/outputs
with torch.no_grad():
    test_sample = tensor_list[0]
    encoded, decoded = models(test_sample)

# convert tensors → numpy arrays for display
orig = test_sample.squeeze().numpy()         # (H, W)
recon = decoded.squeeze().numpy()               # (H, W)
recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)  # normalize to [0,1]
recon = (recon * 255).astype("uint8")
orig = (orig * 255).astype("uint8")

# optional: resize to match (sometimes decoder output is smaller)
if orig.shape != recon.shape:
    recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))

# show both images
cv2.imshow("Original", orig)
cv2.imshow("Reconstructed", recon)
cv2.waitKey(0)
cv2.destroyAllWindows()