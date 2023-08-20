# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:39:30 2023

@author: doguk
"""

import cv2
import os
import numpy as np
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape, Conv2DTranspose, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score

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
    test_images_path = './cifar-10/test'

    images_train = [cv2.imread(os.path.join(train_images_path, image_name + '.png')) for image_name in image_names if os.path.exists(os.path.join(train_images_path, image_name + '.png'))]
    images_test = [cv2.imread(os.path.join(test_images_path, image_name + '.png')) for image_name in image_names if os.path.exists(os.path.join(test_images_path, image_name + '.png'))]

    labels_train = [labels[image_names.index(image_name)] for image_name in image_names if os.path.exists(os.path.join(train_images_path, image_name + '.png'))]
    labels_test = [labels[image_names.index(image_name)] for image_name in image_names if os.path.exists(os.path.join(test_images_path, image_name + '.png'))]

    #Convert lists to numpy arrays
    images_train = np.array(images_train)
    images_test = np.array(images_test)

    #Convert labels to one-hot encoding
    labels_train = pd.get_dummies(labels_train)[unique_labels].values
    labels_test = pd.get_dummies(labels_test)[unique_labels].values

    return images_train, images_test, labels_train, labels_test, unique_labels

def create_classifier(number_of_classes):
    model = Sequential()    
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))  
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    #Optional neuron layer (if needed)
    #model.Add(Dense(128, activation='relu')),
    #model.Add(Dropout(0.2)),
    model.add(Dense(number_of_classes, activation='softmax'))

    return model

def train_classifier(model, images_train, labels_train, images_test, labels_test, epochs, batch_size):
    precision = Precision()
    recall = Recall()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision, recall])
    
    checkpoint = ModelCheckpoint('classifier_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, validation_data=(images_test, labels_test), callbacks=[checkpoint])

    predictions = model.predict(images_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(labels_test, axis=1)
    
    precision_val = precision_score(y_true, y_pred, average='weighted')
    recall_val = recall_score(y_true, y_pred, average='weighted')
    f1_val = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Precision: {precision_val}")
    print(f"Recall: {recall_val}")
    print(f"F1-Score: {f1_val}")
    
def create_generator(base_model):
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.layers[-2].output  # Remove the last dense layer for classification
    x = Reshape((16, 16, 128))(x)  # The dimensions depend on the architecture of your classifier
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    output = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='sigmoid')(x)
    
    generator = Model(base_model.input, output)
    return generator

def create_discriminator():
    image_input = Input(shape=(64, 64, 3))
    label_input = Input(shape=(number_of_classes,))
    label_embedding = Dense(64 * 64 * 3)(label_input)
    label_embedding = Reshape((64, 64, 3))(label_embedding)
    
    merged_input = Concatenate(axis=-1)([image_input, label_embedding])
    
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Flatten()(x)
    validity = Dense(1, activation='sigmoid')(x)
    
    model = Model([image_input, label_input], validity)
    model.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    epochs = int(input("Enter number of epochs for training classifier: "))
    batch_size = int(input("Enter batch size for training classifier: "))
    
    images_train, images_test, labels_train, labels_test, class_labels = load_data()
    number_of_classes = len(class_labels)

    # Save label names in a CSV
    pd.DataFrame(class_labels, columns=["label_names"]).to_csv("label_names.csv", index=False)
    
    # Train the classifier
    classifier = create_classifier(number_of_classes)
    train_classifier(classifier, images_train, labels_train, images_test, labels_test, epochs, batch_size)
    
    # Remaining code for GAN architecture and training as provided before...
    base_model = load_model('classifier_model.h5')
    generator = create_generator(base_model)
    discriminator = create_discriminator()
    
    noise = Input(shape=(100,))
    label = Input(shape=(number_of_classes,))
    
    img = generator([noise, label])
    discriminator.trainable = False
    validity = discriminator([img, label])
    
    combined = Model([noise, label], validity)
    combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    combined.save('combined_model.h5')