# this file is for actually generating a continuation of the gif using autoregression
#it looks very similar to the cnn_test because were essentially doing the same thing but for many images and then AR

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import time

from autoregression import Multimodal_AR
from image_encoders import CNN_Image_Embedding

#loading encoder/decoder model
models = CNN_Image_Embedding(inchannels=1, latentdim=512)
models.load_state_dict(torch.load("cnn_autoencoder.pth"))
models.eval()

#getting the image sequence in greyscale as tensors for the encoder/AR/decoders
frames_folder = os.path.join(os.getcwd(), "image_sequences", "25.gif")  # adjust name as needed
frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(".png")])
tensor_list = []
to_tensor = transforms.ToTensor()
for f in frame_files:
    frame_path = os.path.join(frames_folder, f)
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    img_tensor = to_tensor(frame).unsqueeze(0)
    tensor_list.append(img_tensor)

#generate a new gif
with torch.no_grad():
    #get a list of dicts which are the encoded images from the gif
    encoded_frames = []
    for frame in tensor_list:
        encoded = models.encoder(frame)
        latent_np = encoded.squeeze(0).cpu().numpy()
        latent_mean = np.mean(latent_np)
        latent_std = np.std(latent_np) + 1e-8
        latent_norm = (latent_np - latent_mean) / latent_std
        encoded_frames.append({
            "latent": {str(i+1): float(val) for i, val in enumerate(latent_norm)},
            "mean": latent_mean,
            "std": latent_std
        })

    #run autoregression on the encoded image sequence
    latent_window = [frame["latent"] for frame in encoded_frames]
    AR_dicts = Multimodal_AR.predict_next_window(5, 1, latent_window)

    #convert the autoregression dict back into images list
    recon_list = []
    last_mean = encoded_frames[-1]["mean"]
    last_std = encoded_frames[-1]["std"]
    for pred_dict in AR_dicts:
        latent_array = np.array([pred_dict[str(j+1)] for j in range(len(pred_dict))])
        latent_denorm = latent_array * last_std + last_mean
        latent_tensor = torch.tensor(latent_denorm, dtype=torch.float32).unsqueeze(0)
        reconstructed = models.decoder(latent_tensor)
        recon_list.append(reconstructed)

#visualization of generated frames
os.makedirs("generated_frames", exist_ok=True)
for idx, img_tensor in enumerate(recon_list):
    img = img_tensor.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(f"generated_frames/frame_{idx:03d}.png", img)