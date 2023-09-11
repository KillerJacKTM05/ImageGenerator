# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:27:10 2023

@author: doguk
"""

import numpy as np
import pandas as pd
import cv2
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Concatenate, Reshape, Conv2DTranspose, LeakyReLU

def create_generator(latent_dim, num_classes):
    # Latent space input
    noise_input = Input(shape=(latent_dim,))
    
    # Label input
    label_input = Input(shape=(num_classes,))
    
    # Embedding for categorical input (labels)
    label_embedding = Dense(4 * 4 * 256)(label_input)
    label_embedding = Reshape((4, 4, 256))(label_embedding)
    
    # Foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    noise_layer = Dense(n_nodes)(noise_input)
    noise_layer = LeakyReLU(alpha=0.2)(noise_layer)
    noise_layer = Reshape((4, 4, 256))(noise_layer)
    
    # Merge inputs
    merge = Concatenate()([noise_layer, label_embedding])
    
    # Upsample to 8x8
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Upsample to 16x16
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Upsample to 32x32
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Output layer
    output_img = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    
    # Define and compile the model
    model = Model([noise_input, label_input], output_img)
    
    return model

# Load the generator model
generator = create_generator(100, 10)
generator.load_weights('generator_weights.h5')  # Corrected this line

# Load label names from CSV
label_names = pd.read_csv('label_names.csv')

# Function to generate image based on label
def generate_image(label_index, latent_dim=100):
    # Convert label index to one-hot encoding
    label_onehot = to_categorical([label_index], num_classes=10)
    
    # Generate noise vector
    noise = np.random.normal(0, 1, (1, latent_dim))
    
    # Generate image
    generated_image = generator.predict([noise, label_onehot])
    
    # Rescale image from [-1, 1] to [0, 255]
    generated_image = 0.5 * generated_image + 0.5
    generated_image = np.clip(generated_image * 255, 0, 255).astype(np.uint8)
    
    # Resize the image to 256x256
    generated_image = cv2.resize(generated_image[0], (256, 256))
    
    return generated_image

# Main loop
while True:
    # Get label index from user
    label_index = int(input("Enter a label index between 0 and 9: "))
    
    # Get label name
    label_name = label_names.iloc[label_index]['label_names']
    
    print(f"Generating image for label: {label_name}")
    
    # Generate and save the image
    img = generate_image(label_index)
    filename = f"generated_image_for_{label_name}.png"
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Image saved as {filename}")
    
    # Ask to continue
    cont = input("Do you want to generate another image? (y/n): ")
    if cont.lower() != 'y':
        break