import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image

import os
from tensorflow.python.layers import base

import re

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def CNN():
    
    global target_width
    global target_height

    epochs = 5000

    model = Sequential([
        Conv2D(input_shape=(64, 64, 1), filters=32, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Flatten(),
        # layers.Dense(1024, activation='relu'),
        Dense(800, activation='softmax')
    ])

    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )

    x_data = np.array(train_image_gray).reshape(train_tfrecord_length, target_width, target_height, 1)  #grayscale image
    # x_data = np.array(train_image).reshape(train_tfrecord_length, target_width, target_height, 3)  #original image
    x_data = x_data / 255.  # normalization
    y_data = np.array(train_label_one_hot)



    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2, min_delta=0.0001)
    #history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=64, callbacks=[early_stopping])

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    datagen.fit(x_data)


    X_train, X_test, y_train, y_test = train_test_split(x_data, 
                                                y_data, 
                                                test_size=0.13,
                                                stratify=train_label
                                                )


    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=datagen.flow(X_test, y_test, batch_size=32),
                        validation_steps=len(X_test) / 32,
                        steps_per_epoch=len(X_train) / 32, epochs=epochs, callbacks=[early_stopping])


    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(test_acc)

    model.save('VGG16_dataset.h5')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # plt.plot(acc, 'b', label='Training acc')
    # plt.plot(val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()

    # plt.figure()

    # plt.plot(loss, 'b', label='Training loss')
    # plt.plot(val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()

    # plt.show()

    model.save('CNN.h5')

# def predict(model_name):
#     model = None
    
#     if model_name == "VGG16_TRANSFER.h5":
        
#         # create graph and load weight
#         vgg16 = VGG16(include_top=False, input_shape=(64, 64, 3))
#         model = Sequential(vgg16.layers)

#         for layer in model.layers[:15]:
#             layer.trainable = False

#         model.add(Flatten())
#         model.add(Dense(256, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(class_num, activation='softmax'))

#         model.compile(
#             loss='categorical_crossentropy',
#             optimizer=SGD(lr=1e-4, momentum=0.9),
#             metrics=['accuracy']
#         )
#         model.load_weights(model_name)

#     y_predict = model.predict_classes(x_data)
#     print(len(y_predict))

#     # create csv output
#     y_id = [idx.numpy() for idx in test_idx]    

#     data = list(zip(y_id, y_predict))
#     df = pd.DataFrame(data, columns=['id', 'class'])
#     df.to_csv("test_acc62.csv", index=0)
#     print(df.head())
    
    
#     # verify predict picture
#     idx = 0
#     plt.imshow(test_image[idx])
#     plt.show()
#     print(test_label[idx])
#     print(test_label_one_hot[idx])
#     print(test_idx[idx])
#     print(y_predict[idx])
    
#     return y_predict

def convert_from_tfrecords(filename):
    idx = []
    images = []
    images_gray = []
    labels = []
    label_one_hot = []
    
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_image_dataset = raw_dataset.map(_parse_image_function)
    
    if "train" in filename:
        tfrecord_length = train_tfrecord_length
    else:
        tfrecord_length = test_tfrecord_length
        
    head = 3
    i = 0
    
    for image_features in parsed_image_dataset:
        onehot = np.zeros(class_num)             
        image_raw = image_features['image_raw'].numpy()
        img2 = np.asarray(bytearray(image_raw), dtype="uint8")
        im = cv2.imdecode(img2, cv2.IMREAD_COLOR)
        label = image_features['label']
        id = image_features['id']
        onehot[label] = 1

        if len(im.shape) > 2:
            img_gray = Image.fromarray(im).convert('L')
            img_gray = np.array(img_gray, dtype=np.uint8)

        img_gray = np.expand_dims(img_gray, 2)
        img_gray_3 = np.concatenate((img_gray, img_gray, img_gray),-1) # gray scale to 3 channels
        idx.append(id)
        images_gray.append(img_gray.astype(np.float32))
        images.append(im.astype(np.float32))
        labels.append(label)
        label_one_hot.append(onehot)
        
        # if i < head:
        #     plt.imshow(im)
        #     #plt.imshow(img_gray)
        #     plt.show()
        #     print(img_gray.shape)
        #     print(label)
        #     print(onehot)
        #     print(id)
            
        i += 1
        
    return idx, images, images_gray, labels, label_one_hot

def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


if __name__ == '__main__':
    image_feature_description = {
        'id' : tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    class_num = 800
    train_tfrecord_length = 68804 #55044
    test_tfrecord_length = 13760
    target_width = 64
    target_height = 64
    target_channel = 1

    train_idx, train_image, train_image_gray, train_label, train_label_one_hot = convert_from_tfrecords("datasets/tfrecords/train_data_clean" + str(target_width) + "_" + str(target_height) + ".tfrecords")
    print(len(train_idx))
    train_tfrecord_length = len(train_idx)

    # test_idx, test_image, test_image_gray, test_label, test_label_one_hot = convert_from_tfrecords("test_" + str(target_width) + "_" + str(target_height) + ".tfrecords")
    # print(len(test_idx))

    CNN()
    # x_data = np.array(test_image).reshape(test_tfrecord_length, target_width, target_height, 3)
    # x_data = x_data / 255.  # normalization
    # y_data = np.array(test_label_one_hot)
    # v = predict("VGG16_TRANSFER.h5")
