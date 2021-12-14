import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import sys
import time
import datetime
import tarfile
from PIL import Image
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from IPython.display import display, Image
import shutil


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# Create a dictionary with features that may be relevant.
def image_decode(id, label, image_string):
    feature = {
        'id' : _int64_feature(id),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_index_and_number():
    ori_train_path = 'train/'
    train_list = os.listdir(ori_train_path)

    train_dict = []

    for file in train_list:
        print(file)
        try:
            train_dict.append((file[-5], int(file[0:-6])))
        except Exception as e:
            print(file)
            pass

    print(train_dict[0:10])

    labels = []
    index = []
    for i in range(len(train_dict)):
        labels.append(train_dict[i][0])
        index.append(train_dict[i][1])

    print(labels[0:5])
    print(index[0:5])

    dict = {'label': labels, 'index': index}
    df = pd.DataFrame(dict)
    df.to_csv('train_label_clean.csv', encoding='utf_8_sig')


def count_area_pixel(img, start_pixel, img_high):
    pixel_cnt = 0
    for i in range(start_pixel, start_pixel + 64):
        for j in range(img_high):
            if img[j][i] == 0:
                pixel_cnt += 1
    return pixel_cnt


def search_text_position(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    H, W = thresh.shape
    max_pixel_area = 0
    max_pixel_pos = 0

    for i in range(W - 64):
        tmp = count_area_pixel(thresh, i, H)
        if tmp >= max_pixel_area:
            max_pixel_area = tmp
            max_pixel_pos = i

    img = img[0:H, max_pixel_pos:max_pixel_pos + 64]

    c_H, c_W, channel = img.shape
    top = round(c_H / 7)
    bot = round(c_H / 7 * 6)
    img_top = img[0:top, 0:c_W]
    img_bot = img[bot:c_H, 0:c_W]
    is_line_threshold = 10
    minLineLength = 20
    maxLineGap = 10

    edges = cv2.Canny(img_top, 50, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, is_line_threshold, minLineLength, maxLineGap)

    try:
        lines = lines[:, 0, :]
        #print(len(lines))
        for x1, y1, x2, y2 in lines:
            cv2.line(img_top, (x1, y1), (x2, y2), (255, 255, 255), 3)
    except Exception as e:
        pass
        #print("This top img didn't detect any line")

    edges = cv2.Canny(img_bot, 50, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, is_line_threshold, minLineLength, maxLineGap)

    try:
        lines = lines[:, 0, :]
        #print(len(lines))
        for x1, y1, x2, y2 in lines:
            cv2.line(img_bot, (x1, y1), (x2, y2), (255, 255, 255), 3)
    except Exception as e:
        pass
        #print("This bottom img didn't detect any line")

    return img


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Create a dictionary with features that may be relevant.
def image_decode(id, label, image_string):
    feature = {
        'id': _int64_feature(id),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_to_tfrecords():
    df = pd.read_csv('train_label_clean.csv', encoding='utf_8_sig')
    print(df[0:5])

    f = open('training data dic.txt', encoding='utf_8_sig')
    words = f.read().splitlines()
    print(words[0:5])

    df['label_idx'] = None
    print(df[0:5])

    for i in range(len(df['label'])):
        word = df['label'][i]
        if not word in words:
            df['label_idx'][i] = -1
        else:
            idx = words.index(word)
            df['label_idx'][i] = idx

    print(df[0:10])

    print(min(df['label_idx']))
    print(max(df['label_idx']))


    data_path = 'train/'

    images = glob.glob(data_path + '*.jpg')

    tfrecord_filename = 'datasets/tfrecords/train_data_clean' + str(width_re) + '_' + str(height_re) + '.tfrecords'
    writer = tf.compat.v1.python_io.TFRecordWriter(tfrecord_filename)

    print(images[0:5])


    data_augmentation = tf.keras.Sequential([
        # layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.05),
        # training will do normalization, so don't do it here
        # layers.experimental.preprocessing.Rescaling(1./255).
    ])

    img_count = 0

    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for image in images:
            img = cv2.imread(image)

            # resize
            img_re = cv2.resize(img, (width_re, height_re))
            # focus on text
            img_re = search_text_position(img_re)


            id = int(image.split('/')[-1].split('_')[0])
            mask = df['index'] == id
            label = df[mask]['label_idx']
            label = label.iloc[0]
            if label == -1:
                continue
            image_string = cv2.imencode('.jpg', img_re)[1].tobytes()
            tf_decode = image_decode(id, label, image_string)
            writer.write(tf_decode.SerializeToString())
            img_count += 1

            # data augumentation, pictures total to 206412
            img_re_aug = tf.expand_dims(img_re, 0)

            for i in range(2):
                augmented_image = data_augmentation(img_re_aug)
                image_string = cv2.imencode('.jpg', augmented_image[0].numpy())[1].tobytes()

                tf_decode = image_decode(id, label, image_string)
                writer.write(tf_decode.SerializeToString())
                img_count += 1


    # expect 68804 pictures for ori
    print(img_count)


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def convert_from_tfrecords():
    raw_dataset = tf.data.TFRecordDataset('train_data_clean' + str(width_re) + '_' + str(height_re) + '.tfrecords')
    parsed_image_dataset = raw_dataset.map(_parse_image_function)

    i = 0
    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        img2 = np.asarray(bytearray(image_raw), dtype="uint8")
        im = cv2.imdecode(img2, cv2.IMREAD_COLOR)
        plt.imshow(im)
        plt.show()
        i += 1
        if i == 5:
            break


if __name__ == '__main__':

    try:
        shutil.rmtree('train')
    except Exception as e:
        pass

    try:
        file_name = sys.argv[1]
    except IndexError:
        file_name = 'esun.tar'

    if file_name == '-h':
        print("usage: python preprocessing.py [tar] \t\t\t>>>looking for tar file in the path "
              "of datasets/raw_data/esun_images/")
        print("example:\ntime python preprocessing.py \t\t\t\t>>>default tar is "
              "datasets/raw_data/esun_images/esun.tar which is not filtered")
        print("time python preprocessing.py esun_less.tar \t\t>>>link to datasets/raw_data/esun_images/esun_less.tar"
              ", which is few test pictures")
        exit()

    #init settings
    width_re, height_re = 64, 64
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    tar = tarfile.open('datasets/raw_data/esun_images/' + file_name, "r:")
    tar.extractall()
    tar.close()

    if file_name != 'esun.tar':
        shutil.move(file_name[0:-4], 'train/')

    save_index_and_number()
    convert_to_tfrecords()

    #show image from tfrecords
    #convert_from_tfrecords()



