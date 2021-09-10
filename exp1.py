import os
import cv2
import math
import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.python.ops.gen_array_ops import pad
from tensorflow.python.ops.gen_batch_ops import batch
from generator import DataGenerator

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

image_height, image_width = 224, 224
max_images_per_class = 800

dataset_directory = "UCF50"
classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

model_output_size = len(classes_list)

def plot_samples():
    plt.figure(figsize = (30, 30))

    all_classes_names = os.listdir('UCF50')
    random_range = random.sample(range(len(all_classes_names)), 20)

    for counter, random_index in enumerate(random_range, 1):
        selected_class_Name = all_classes_names[random_index]
        video_files_names_list = os.listdir(f'UCF50/{selected_class_Name}')
        selected_video_file_name = random.choice(video_files_names_list)
        video_reader = cv2.VideoCapture(f'UCF50/{selected_class_Name}/{selected_video_file_name}')
        _, bgr_frame = video_reader.read()
        video_reader.release()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
        plt.subplot(5, 4, counter)
        plt.imshow(rgb_frame)
        plt.axis('off')

def frames_extraction(video_path):
    
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)

    while True:
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list


def create_model():
    # model = Sequential()

    # model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(image_height, image_width, 3), padding='same'))
    # model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))


    # model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Dense(4096, activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(model_output_size, activation = 'softmax'))
    # model.summary()
    # return model


    model = Sequential()

    model.add(Conv2D(input_shape=(image_height,image_width,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=model_output_size, activation="softmax"))
    
    model.summary()
    return model

     # We will use a Sequential model for model construction
    # model = Sequential()

    # # Defining The Model Architecture
    # model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)))
    # model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size = (2, 2)))
    # model.add(GlobalAveragePooling2D())
    # model.add(Dense(256, activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(model_output_size, activation = 'softmax'))

    # # Printing the models summary
    # model.summary()

    # return model


def create_dataset():

    # Declaring Empty Lists to store the features and labels values.
    temp_features = [] 
    features = []
    labels = []
    
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        
        files_list = os.listdir(os.path.join(dataset_directory, class_name))
        for file_name in files_list:

            video_file_path = os.path.join(dataset_directory, class_name, file_name)
            frames = frames_extraction(video_file_path)
            temp_features.extend(frames)
        features.extend(random.sample(temp_features, max_images_per_class))
        labels.extend([class_index] * max_images_per_class)
        temp_features.clear()

    features = np.asarray(features)
    labels = np.array(labels)  

    return features, labels


# Main =========

model = create_model()

features, labels = create_dataset()
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.2, shuffle = True, random_state = seed_constant)

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)
print("model compile==================")
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
print("modle fit =====================")
# model_training_history = model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 16 , shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])

train_gen = DataGenerator(features_train, labels_train, dim=(image_width, image_height, 3), n_classes=4)
test_gen = DataGenerator(features_test, labels_test, dim=(image_height, image_width, 3), n_classes=4)

model_training_history = model.fit_generator(generator=train_gen,
                    validation_data=test_gen,
                    use_multiprocessing=True,
                    epochs=50,
                    batch_size= 16,
                    workers=6)

