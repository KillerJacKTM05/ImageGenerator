# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:34:54 2023

@author: doguk
"""

import cv2
import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from keras.layers import Conv2D, Flatten, Dense, Dropout, Input, Concatenate, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.compat.v1.keras.backend import set_session
from sklearn.model_selection import train_test_split

def load_data():
    #Loading the labels and image names from CSV, its format like this: "1" "Car", "2" "Plane", ...
    df = pd.read_csv('./cifar-10/trainLabels.csv')
    #print(df.head())
    
    #Extracting the only column name which holds the labels
    label_column = df.columns[1]    
    #Extract labels and their corresponding image names
    image_column = df.columns[0]
    
    labels = df[label_column].tolist()
    image_names = df[image_column].astype(str).tolist()
    unique_labels = df[label_column].unique().tolist()

    #Load train and test images based on image names
    train_images_path = './cifar-10/train'

    images_train = [cv2.imread(os.path.join(train_images_path, image_name + '.png')) for image_name in image_names]
    images_train = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_train]
  
    labels_train = [labels[image_names.index(image_name)] for image_name in image_names if os.path.exists(os.path.join(train_images_path, image_name + '.png'))]

    #Convert labels to one-hot encoding
    labels_train = pd.get_dummies(labels_train)[unique_labels].values
    images_train = normalize_data(images_train)

    return images_train, labels_train, unique_labels

def normalize_data(images):
    images = np.array(images, dtype=np.float32) # Convert list to numpy array
    return images / 127.5 - 1

# Residual Block
def res_block(x, filters, kernel_size=3, stride=1, padding='same', activation='relu'):
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x
    
# Updated Generator Model
def create_generator(latent_dim, num_classes):
    noise_input = Input(shape=(latent_dim,))
    labels_input = Input(shape=(num_classes,))

    labels_embedding = Dense(4 * 4 * 128)(labels_input)
    labels_embedding = Reshape((4, 4, 128))(labels_embedding)

    noise_layer = Dense(4 * 4 * 128)(noise_input)
    noise_layer = Reshape((4, 4, 128))(noise_layer)

    merged_input = Concatenate()([noise_layer, labels_embedding])

    # Downsampling
    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(merged_input)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    filter = 256
    # Residual blocks
    for _ in range(6):
        x = res_block(x, filter, 3, (3,3))
    
    # Upsampling
    x = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.75)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.75)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(x)
    output_img = layers.Activation('tanh')(x)
    
    model = Model([noise_input, labels_input], output_img)
    return model

# Updated Discriminator Model
def create_discriminator(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(512, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(1024, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def train_gan(gen, disc, comb, epochs, batch_size, images_t, labels_t, num_of_classes):
    print("starting..")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    #plot metrics
    epoch_d_loss = []
    epoch_g_loss = []
    
    # The loss function and optimizers
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    d_loss_metric = tf.keras.metrics.Mean()
    g_loss_metric = tf.keras.metrics.Mean()

    # Split dataset into batches
    dataset = tf.data.Dataset.from_tensor_slices((images_t, labels_t)).shuffle(buffer_size=1024).batch(batch_size)
    #dataset = dataset.batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        for idx, (real_images, batch_labels) in tqdm(enumerate(dataset), desc=f"Epoch {epoch + 1}", ncols=100):
            if real_images.shape[0] != batch_size:
                continue  #no need for now anyway we are shuffling
                
            batch_labels = labels_t[idx * batch_size: (idx + 1) * batch_size]
            # Calculate batch size dynamically
            batch_size = tf.shape(real_images)[0]

            # Produce random vectors in the latent space
            random_latent_vectors = tf.random.normal(shape=(batch_size, 100))  # Assuming latent_dim=100
            
            # Fake images generation
            generated_images = gen([random_latent_vectors, batch_labels])

            # Labels for the discriminator
            real_labels = tf.zeros((batch_size, 1))
            fake_labels = tf.ones((batch_size, 1))
            labels = tf.concat([real_labels, fake_labels], axis=0)
            labels += 0.05 * tf.random.uniform(tf.shape(labels))

            combined_images = tf.concat([real_images, generated_images], axis=0)

            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = disc(combined_images)
                d_loss = loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, disc.trainable_weights)
            d_optimizer.apply_gradients(zip(grads, disc.trainable_weights))

            # Train the generator
            misleading_labels = tf.zeros((batch_size, 1))
            with tf.GradientTape() as tape:
                predictions = disc(gen([random_latent_vectors, batch_labels]))
                g_loss = loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, gen.trainable_weights)
            g_optimizer.apply_gradients(zip(grads, gen.trainable_weights))

            # Update metrics
            d_loss_metric.update_state(d_loss)
            g_loss_metric.update_state(g_loss)
            epoch_d_loss.append(d_loss_metric)
            epoch_g_loss.append(g_loss_metric)

        print(f"Epoch {epoch + 1} completed. D Loss: {d_loss_metric.result()} | G Loss: {g_loss_metric.result()}")
        save_models(gen, disc, comb)
        # Use GANMonitor for visualization
        monitor = SimpleGANMonitor(generator=gen, num_img=1, latent_dim=100, num_classes=number_of_classes)
        monitor.on_epoch_end(epoch)
    
    print("Training Finished..")    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_g_loss, label='Generator Loss')
    plt.plot(epoch_d_loss, label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

class SimpleGANMonitor:
    def __init__(self, generator, num_img=1, latent_dim=100, num_classes=10):
        self.generator = generator
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def on_epoch_end(self, epoch):
        # Randomly sample labels for generating images
        random_labels = np.random.randint(0, self.num_classes, self.num_img)
        random_labels_one_hot = to_categorical(random_labels, self.num_classes)

        # Generate noise vector
        noise = np.random.normal(0, 1, (self.num_img, self.latent_dim))

        # Generate images
        generated_images = self.generator.predict([noise, random_labels_one_hot])

        # Rescale images from [-1, 1] to [0, 255]
        generated_images = 0.5 * generated_images + 0.5
        generated_images = np.clip(generated_images * 255, 0, 255).astype(np.uint8)

        # Save images
        for i, label in enumerate(random_labels):
            filename = f"generated_image_label_{class_labels[label]}_epoch_{epoch+1}.png"
            cv2.imwrite(filename, cv2.cvtColor(generated_images[i], cv2.COLOR_RGB2BGR))

def save_models(generator, discriminator, combined):
    generator.save_weights('generator_weights.h5')
    discriminator.save_weights('discriminator_weights.h5')
    combined.save_weights('combined_weights.h5')
    
if __name__ == "__main__":
    #Allow gpu options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    gan_epochs = int(input("Enter number of epochs for training GAN: "))
    batch_size = int(input("Enter batch size for training GAN: "))
    
    images_train, labels_train, class_labels = load_data()
    number_of_classes = len(class_labels)
    #split the labelled images as 90% training 10% validation
    images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=0.10, random_state=42)
    
    #Save label names in a CSV
    pd.DataFrame(class_labels, columns=["label_names"]).to_csv("label_names.csv", index=False)
    
    #Create models
    generator = create_generator(100, number_of_classes)
    discriminator = create_discriminator()
    #Load if exists
    if os.path.exists('generator_weights.h5'):
        generator.load_weights('generator_weights.h5')
        print("Loaded generator weights.")

    if os.path.exists('discriminator_weights.h5'):
        discriminator.load_weights('discriminator_weights.h5')
        print("Loaded discriminator weights.")      
   
    noise = Input(shape=(100,))
    labels_input = Input(shape=(number_of_classes,))
    generated_images = generator([noise, labels_input])
    validity = discriminator(generated_images)

    combined = Model([noise, labels_input], validity)
    optimizer_com = tf.keras.optimizers.Adam(learning_rate = 0.00002)
    combined.compile(optimizer=optimizer_com, loss='binary_crossentropy')
    train_gan(generator, discriminator, combined, gan_epochs, batch_size, images_train, labels_train, number_of_classes)
