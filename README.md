# Human Image Background Remover

## Overview

This project implements a deep learning model for human image background removal using a U-Net architecture. The model is trained using TensorFlow and Keras on a dataset containing human images and their corresponding segmentation masks.

## Dataset

The dataset used for training was sourced from Kaggle:
[Segmentation Full Body MADS Dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset)

### Important Notes on the Dataset

- The dataset consists of images with very similar backgrounds.
- The human shapes in the images are consistent in size and structure.
- Due to these constraints, the model does not perform well on real-world images that have complex backgrounds and varied human shapes.
- This project serves as a conceptual and practice approach rather than a fully generalized background removal solution.

## Model Architecture

The model follows a U-Net architecture:

- Downsampling path using convolutional and max-pooling layers.
- A bottleneck layer with high-level feature extraction.
- Upsampling path using transposed convolutions and skip connections to retain details.
- Final output layer with a sigmoid activation function to generate the segmentation mask.

## Training Process

- The dataset is preprocessed by resizing images to 256x256 and normalizing pixel values.
- A batch size of 32 is used for training.
- The model is trained for 50 epochs using the Adam optimizer and binary cross-entropy loss.
- A custom callback saves the model every 5 epochs.

## Usage

1. Clone the repository and navigate to the project directory.
2. Install dependencies: `pip install tensorflow matplotlib numpy`
3. Run `train.ipynb` in Google Colab or locally to train the model.

## Results

- The model successfully removes backgrounds from images within the training dataset.
- However, its performance is limited on real-world images due to dataset constraints.

## Future Improvements

- Training on a more diverse dataset with complex backgrounds.
- Fine-tuning the model with additional real-world images.
- Exploring advanced segmentation techniques for better generalization.