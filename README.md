# Autoregression-GIF-Generation
This repository is used for testing an autoregression based GIF generator. The generator uses a CNN model to encode the GIF's image sequence before running autoregression across the feature vector, then uses the same CNN structure to decode the resulting image sequence back into an image sequence.

## About
This thing doesn't really work at present as the dataset I use for the CNN encoder/decoder are WAY too small, its like ~500 images or something. In the future when I come back to this I'll collect more gif's to use for training, but right now this was just to test the architecture for other projects that use a similar structure.

The way it works is that it's converting the image sequence used into 64x64 pixel greyscale images, then using those to train the CNN's. It then uses the encoder CNN to turn each image into a feature vector and do that for a sequence of images in order. It then uses the multimodal-autoregression across the encoded sequence to attempt a prediction at the next n frames, which is set to 1 currently. It then takes the output which is the predicted feature vector and use the decoder CNN to turn it back into a 64x64 greyscale image. 

To run it for yourself, run main.py, it will output the resulting image sequence to the folder called generated_frames.

## Libraries-Used

- Standard python libraries such as os, time or PIL for standard functions
- Numpy for math stuff
- OpenCV for image manipulation and gif image separation
- PyTorch for creating, training and using the encoder/decoder
