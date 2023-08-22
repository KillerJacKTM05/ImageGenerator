# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:39:30 2023

@author: doguk
"""

import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization
from keras.losses import MeanSquaredError
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def load_data():
    #Loading the labels and image names from CSV, its format like this: "1" "Car", "2" "Plane", ...
    df = pd.read_csv('./cifar-10/trainLabels.csv')
    
    #Extracting the only column name which holds the labels
    label_column = df.columns[1]    
    #Extract labels and their corresponding image names
    image_column = df.columns[0]
    
    labels = df[label_column].tolist()
    image_names = df[image_column].astype(str).tolist()
    unique_labels = df[label_column].unique().tolist()

    #Load train and test images based on image names
    train_images_path = './cifar-10/train'
    #test_images_path = './cifar-10/test'

    images_train = [cv2.imread(os.path.join(train_images_path, image_name + '.png')) for image_name in image_names]
    images_train = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_train]
    #images_test = [cv2.imread(os.path.join(test_images_path, image_name + '.png')) for image_name in image_names if os.path.exists(os.path.join(test_images_path, image_name + '.png'))]

    labels_train = [labels[image_names.index(image_name)] for image_name in image_names if os.path.exists(os.path.join(train_images_path, image_name + '.png'))]
    #labels_test = [labels[image_names.index(image_name)] for image_name in image_names if os.path.exists(os.path.join(test_images_path, image_name + '.png'))]

    #Convert lists to numpy arrays
    images_train = np.array(images_train).astype('float32') / 255.0
    #images_test = np.array(images_test)

    #Convert labels to one-hot encoding
    labels_train = pd.get_dummies(labels_train)[unique_labels].values
    #labels_test = pd.get_dummies(labels_test)[unique_labels].values

    return images_train, labels_train, unique_labels

def create_classifier(number_of_classes):
    model = Sequential()    
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))  
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    #Optional neuron layer (if needed)
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))

    return model

def visualize_sample(images, labels, unique_labels):
    fig, axes = plt.subplots(1, 4, figsize=(20,5))
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img)
        ax.set_title(unique_labels[np.argmax(label)])
        ax.axis('off')
    plt.show()
    
def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy.
    """
    fig, axs = plt.subplots(2)
    
    # Plot training & validation accuracy values
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show() 
    
def train_classifier(model, images_train, labels_train, epochs, batch_size, validation_data):
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', precision, recall])
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0004, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('classifier_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(images_train)

    history = model.fit_generator(
        datagen.flow(images_train, labels_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=validation_data,  
        callbacks=[checkpoint, reduce_lr]
    )

    images_val, labels_val = validation_data
    predictions = model.predict(images_val)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(labels_val, axis=1)

    precision_val = precision_score(y_true, y_pred, average='weighted')
    recall_val = recall_score(y_true, y_pred, average='weighted')
    f1_val = f1_score(y_true, y_pred, average='weighted')
    
    plot_training_history(history)
    print(f"Precision: {precision_val}")
    print(f"Recall: {recall_val}")
    print(f"F1-Score: {f1_val}")

def feature_extractor_from_classifier(classifier_model):
    # Discard the last layer (output classification layer)
    model = Sequential(classifier_model.layers[:-1])

    # Make the layers non-trainable (optional, but recommended to prevent interference with GAN training)
    for layer in model.layers:
        layer.trainable = False

    return model
  
def save_generated_images(generator, epoch, number_of_classes, examples=2, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, [examples, 100])
    # Generate random labels for the generator
    sampled_labels = np.random.randint(0, number_of_classes, examples).reshape(-1, 1)
    sampled_labels = tf.keras.utils.to_categorical(sampled_labels, number_of_classes)
    generated_images = generator.predict([noise, sampled_labels])
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
    
def create_generator(latent_dim, num_classes):
    # Noise input
    noise_input = Input(shape=(latent_dim,))
    
    # Label input
    labels_input = Input(shape=(num_classes,))
    labels_embedding = Dense(4 * 4 * 128, activation="relu")(labels_input)
    labels_embedding = Reshape((4, 4, 128))(labels_embedding)

    # Process noise
    noise_layer = Dense(4 * 4 * 128, activation='relu')(noise_input)
    noise_layer = Reshape((4, 4, 128))(noise_layer)

    # Merge noise and label embeddings
    merged_input = Concatenate()([noise_layer, labels_embedding])

    # Continue with the rest of your generator architecture
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(merged_input)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    output_img = Conv2DTranspose(3, (3, 3), padding='same', activation='sigmoid')(x)

    model = Model([noise_input, labels_input], output_img)

    return model

def create_discriminator(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def train_gan(generator, discriminator, combined, feature_extractor, epochs, batch_size, images_train, labels_train, number_of_classes):
    mse_loss = MeanSquaredError()
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        idx = np.random.randint(0, images_train.shape[0], batch_size)
        imgs = images_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate random labels to guide the generator
        sampled_labels = np.random.randint(0, number_of_classes, batch_size).reshape(-1, 1)
        sampled_labels = tf.keras.utils.to_categorical(sampled_labels, number_of_classes)

        gen_imgs = generator.predict([noise, sampled_labels])

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Feature matching loss
        real_features = feature_extractor.predict(imgs)
        fake_features = feature_extractor.predict(gen_imgs)
        feature_loss = mse_loss(real_features, fake_features)

        # Consider the feature loss when training the generator
        g_loss = combined.train_on_batch([noise, sampled_labels], valid) + feature_loss

        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
        if epoch % 500 == 0:
            save_generated_images(generator, epoch, number_of_classes)
        
    generator.save("generator_model.h5")



if __name__ == "__main__":
    epochs = int(input("Enter number of epochs for training classifier: "))
    batch_size = int(input("Enter batch size for training classifier: "))
    
    images_train, labels_train, class_labels = load_data()
    number_of_classes = len(class_labels)
    #Visualize random samples
    indices = np.random.choice(range(images_train.shape[0]), 4, replace=False)
    visualize_sample(images_train[indices], labels_train[indices], class_labels)
    #split the labelled images as 90% training 10% validation
    images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=0.10, random_state=42)
    
    #Save label names in a CSV
    pd.DataFrame(class_labels, columns=["label_names"]).to_csv("label_names.csv", index=False)
    
    #Check first training completed
    checkTrain = input("Do you want to skip training the classifier? (y/n)")
    if (checkTrain == "y" or checkTrain == "Y"):
        print("Skipped...")
    else:
        #Train the classifier
        if (os.path.exists('./classifier_model.h5')):
            classifier = load_model('classifier_model.h5')
            print("existing model found and loaded.")
        else:
            classifier = create_classifier(number_of_classes)
            print("new model instance created.")

        train_classifier(classifier, images_train, labels_train, epochs, batch_size, (images_val, labels_val))       
    
    base_model = load_model('classifier_model.h5')
    feature_extractor = feature_extractor_from_classifier(base_model)
    generator = create_generator(100, number_of_classes)
    discriminator = create_discriminator()
    
    noise = Input(shape=(100,))
    labels_input = Input(shape=(number_of_classes,))
    generated_images = generator([noise, labels_input])
    validity = discriminator(generated_images)

    combined = Model([noise, labels_input], validity)
    optimizer_com = tf.keras.optimizers.Adam(learning_rate = 0.0005)
    combined.compile(optimizer=optimizer_com, loss='binary_crossentropy')
    train_gan(generator, discriminator, combined, feature_extractor,epochs, batch_size, images_train, labels_train, number_of_classes)
