import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from image_encoders import CNN_Image_Embedding
from PIL import Image

# --- CONFIG ---
image_root = os.path.join(os.getcwd(), "image_sequences")
batch_size = 32
epochs = 25
latent_dim = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET ---
class ImageFolderDataset(Dataset):
    def __init__(self, root_folder):
        self.paths = []
        for root, _, files in os.walk(root_folder):
            for f in files:
                if f.lower().endswith(".png"):
                    self.paths.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Grayscale(),     # ensure 1 channel
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

# --- LOAD DATA ---
print("Loading Data...")
dataset = ImageFolderDataset(image_root)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- MODEL ---
print("Loading Model...")
model = CNN_Image_Embedding(inchannels=1, latentdim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

# --- TRAIN LOOP ---
print("Starting Training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs in dataloader:
        imgs = imgs.to(device)

        optimizer.zero_grad()
        z, recon = model(imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

# --- SAVE TRAINED MODEL ---
torch.save(model.state_dict(), "cnn_autoencoder.pth")
print("Training complete and model saved as cnn_autoencoder.pth")
