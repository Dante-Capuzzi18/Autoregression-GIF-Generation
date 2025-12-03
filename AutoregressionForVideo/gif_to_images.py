#This file is to break a gif into multiple images as pngs, saved to image_sequences/gifname/frame(1-n)

import os
import cv2

#use the name of the gif here
gif_folder = os.path.join(os.getcwd(), "gifs")
output_folder = os.path.join(os.getcwd(), "image_sequences")
gif_files = [f for f in os.listdir(gif_folder) if f.lower().endswith(".gif")]
for gif_name in gif_files:
    gif_path = os.path.join(gif_folder, gif_name)
    cap = cv2.VideoCapture(gif_path)
    os.makedirs(os.path.join(output_folder, gif_name), exist_ok=True)
    
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(os.path.join(output_folder, gif_name), f"frame_{i:03d}.png")
        resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(frame_path, gray)
        i += 1
        
        cap.release()