import numpy as np
import random

import os
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL import ImageFont
from PIL.ImageFont import truetype

from skimage.util import random_noise
from skimage import img_as_float
from math import *
import cv2
import datetime, time
from tensorflow.python.ops import summary_ops_v2

import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

width_re, height_re = 64, 64
num_classes = 800
train_tfrecord_length = 55044 #68804 #
test_tfrecord_length = 13760
target_channel = 1


image_feature_description = {
    'id' : tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

class ocr_network(object):
    def __init__(self, birnn_type=2, lstm_num_layers=1, batch_size=32, max_timestep=None):
        self.save_dir = "./models"

        self.table = []
        for i in range(256):
            self.table.append(i * 1.97)

        # Variables
        self.char_nums = 1
        self.filters_size = [32, 64, 128, 128]    # or [32, 64, 128, 256], [64, 128, 256, 512]
        self.cnn_layer_num = 4
        self.batch_size = batch_size
        self.channels = 1
        self.img_height = height_re
        self.img_width = width_re

        if max_timestep is None:
            pass
            # self.max_timestep = tf.placeholder(tf.int32, name="max_timestep")
        else:
            self.max_timestep = max_timestep
        self.num_hidden = 128  # or 256 , 512

        self.lstm_num_layers = lstm_num_layers
        # 1:static_bidirectional_rnn  2:bidirectional_dynamic_rnn
        # 3:stack_bidirectional_dynamic_rnn  4:stack_bidirectional_rnn
        # 0:dynamic_rnn
        self.birnn_type = birnn_type
        self.num_classes=800

        self.inputs_ = tf.keras.layers.Input(shape=[self.img_height, None, self.channels], dtype="float32", name='inputs')

        with tf.name_scope('cnn'):
            self.layer = self.inputs_
            for i in range(self.cnn_layer_num):
                with tf.name_scope('cnn_layer-%d' % i):
                    self.layer = self.cnn_layer(self.layer, self.filters_size[i])
                    print(self.layer.get_shape())

        _, feature_h, feature_w, cnn_out_channels = self.layer.get_shape().as_list()
        with tf.name_scope('lstm'):

            if self.birnn_type == 0:  # dynamic_rnn
                # [batch_size, feature_w, feature_h, cnn_out_channels]
                self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                # `feature_w` is max_timestep in lstm.  # feature_w(self.max_timestep) unknown
                self.layer = tf.keras.layers.Reshape([-1, feature_h * cnn_out_channels])(self.layer)

                print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
                self.cell1 = tf.keras.layers.LSTMCell(self.num_hidden)
                self.cell2 = tf.keras.layers.LSTMCell(self.num_hidden)

                # [batch_size, max_timestep, self.num_hidden]
                outputs = tf.keras.layers.RNN([self.cell1, self.cell2], return_sequences=True)(self.layer)

                self.logits = tf.keras.layers.Dense(self.num_classes)(outputs)

                # Time major  [max_timestep, batch_size, num_classes]
                self.logits = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.logits)

                self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                self.model.summary()
            elif self.birnn_type == 1 and max_timestep is not None:  # static_bidirectional_rnn
                # [batch_size, feature_w, feature_h, cnn_out_channels]
                self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                # `feature_w` is max_timestep in lstm.
                # [batch_size, max_timestep, feature_h * cnn_out_channels]
                self.layer = tf.keras.layers.Reshape([self.max_timestep, feature_h * cnn_out_channels])(self.layer)

                print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                self.cells = [tf.keras.layers.LSTMCell(self.num_hidden) for _ in range(self.lstm_num_layers)]
                self.cells_stack = tf.keras.layers.StackedRNNCells(self.cells)
                self.rnn_cells_stack = tf.keras.layers.RNN(self.cells_stack, return_sequences=True)

                # layer: [max_timestep, batch_size, feature_h * cnn_out_channels]
                self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.layer)

                # outputs: [max_timestep, batch_size, 2*num_hidden]
                outputs = tf.keras.layers.Bidirectional(self.rnn_cells_stack)(self.layer)

                # Time major  [max_timestep, batch_size, num_classes]
                self.logits = tf.keras.layers.Dense(self.num_classes)(outputs)

                self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                self.model.summary()
            elif self.birnn_type == 2:  # bidirectional_dynamic_rnn
                # [batch_size, feature_w, feature_h, cnn_out_channels]
                self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                # `feature_w` is max_timestep in lstm.
                # [batch_size, max_timestep, feature_h * cnn_out_channels]
                self.layer = tf.keras.layers.Reshape([-1, feature_h * cnn_out_channels])(self.layer)

                print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                self.cells = [tf.keras.layers.LSTMCell(self.num_hidden) for _ in range(self.lstm_num_layers)]
                self.cells_stack = tf.keras.layers.StackedRNNCells(self.cells)
                self.rnn_cells_stack = tf.keras.layers.RNN(self.cells_stack, return_sequences=True)

                # [batch_size, max_timestep, 2 * num_hidden]
                outputs = tf.keras.layers.Bidirectional(self.rnn_cells_stack)(self.layer)

                # [batch_size, max_timestep, num_classes]
                self.logits = tf.keras.layers.Dense(self.num_classes)(outputs)

                # Time major  [max_timestep, batch_size, num_classes]
                self.logits = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.logits)

                self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                self.model.summary()
            elif self.birnn_type == 3:  # stack_bidirectional_dynamic_rnn
                # [batch_size, feature_w, feature_h, cnn_out_channels]
                self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                # `feature_w` is max_timestep in lstm.
                # [batch_size, max_timestep, feature_h * cnn_out_channels]
                self.layer = tf.keras.layers.Reshape([-1, feature_h * cnn_out_channels])(self.layer)

                print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                # [batch_size, max_timestep, 2 * num_hidden]
                for _ in range(self.lstm_num_layers):
                    self.lstm_cell = tf.keras.layers.LSTM(self.num_hidden, return_sequences=True)
                    self.layer = tf.keras.layers.Bidirectional(self.lstm_cell)(self.layer)

                # [batch_size, max_timestep, num_classes]
                self.logits = tf.keras.layers.Dense(self.num_classes)(self.layer)

                # Time major  [max_timestep, batch_size, num_classes]
                self.logits = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.logits)

                self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                self.model.summary()
            elif self.birnn_type == 4 and max_timestep is not None:  # stack_bidirectional_rnn
                # [batch_size, feature_w, feature_h, cnn_out_channels]
                self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [0, 2, 1, 3]))(self.layer)

                # `feature_w` is max_timestep in lstm.
                # [batch_size, max_timestep, feature_h * cnn_out_channels]
                self.layer = tf.keras.layers.Reshape([self.max_timestep, feature_h * cnn_out_channels])(self.layer)

                print('lstm input shape: {}'.format(self.layer.get_shape().as_list()))

                # layer: [max_timestep, batch_size, feature_h * cnn_out_channels]
                self.layer = tf.keras.layers.Lambda(lambda layer: tf.transpose(layer, [1, 0, 2]))(self.layer)

                # outputs: [max_timestep, batch_size, 2*num_hidden]
                for _ in range(self.lstm_num_layers):
                    self.lstm_cell = tf.keras.layers.LSTM(self.num_hidden, return_sequences=True)
                    self.layer = tf.keras.layers.Bidirectional(self.lstm_cell)(self.layer)

                # Time major  [max_timestep, batch_size, num_classes]
                self.logits = tf.keras.layers.Dense(self.num_classes)(self.layer)

                self.model = tf.keras.Model(inputs=[self.inputs_], outputs=[self.logits])

                self.model.summary()


        self.optimizer = tf.keras.optimizers.Adam(1e-3)  # decay=0.98

        if tf.io.gfile.exists(self.save_dir):
            pass

        else:
            tf.io.gfile.makedirs(self.save_dir)

        train_dir = os.path.join(self.save_dir, 'summaries', 'train')
        test_dir = os.path.join(self.save_dir, 'summaries', 'eval')



        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints_biLstm')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        self.avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.avg_acc = tf.keras.metrics.Mean('acc', dtype=tf.float32)
        self.avg_label_error_rate = tf.keras.metrics.Mean('ler', dtype=tf.float32)


    def cnn_layer(self, layer, filter_size):
        layer = tf.keras.layers.Conv2D(kernel_size=3, filters=filter_size, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, fused=True)(layer)
        layer = tf.keras.layers.ELU()(layer)
        layer = tf.keras.layers.MaxPool2D(strides=2, padding="same")(layer)

        return layer

    def save(self, in_global_step=None):
        self.checkpoint.save(self.checkpoint_prefix)
        self.model.save('ocr_model')
        # save_path = self.saver.save(self.sess, os.path.join(self.save_dir, 'best_model.ckpt'),
        #                             global_step=in_global_step)    #self.global_step
        print("Model saved in file: {}".format(self.checkpoint_prefix))

    def imgaug_process(self, data):
        # do something here
        return np.array(data, dtype=np.uint8)
    
    def accuracy_calculation(self, original_seq, decoded_seq, ignore_value=-1, isPrint=False):
        # print(original_seq.shape)
        # print(decoded_seq.shape)

        if len(original_seq) != len(decoded_seq):
            print('original lengths({}) is different from the decoded_seq({}), please check again'.format(len(original_seq), len(decoded_seq)))
            return 0
        count = 0
        for i, origin_label in enumerate(original_seq):
            decoded_label = [j.numpy() for j in decoded_seq[i] if j.numpy() != ignore_value]
            if isPrint:
                print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

                with open('./test.csv', 'w') as f:
                    f.write(str(origin_label) + '\t' + str(decoded_label))
                    f.write('\n')

            try:
                if len(origin_label) == len(decoded_label):
                    match_num = 0
                    for i, c in enumerate(origin_label):
                        if decoded_label[i] == c:
                            match_num += 1
                    if match_num == len(decoded_label):
                        count += 1
            except:
                pass
        return count * 1.0 / len(original_seq)

    def compute_loss(self, labels, logits, seq_len):

        with tf.name_scope("loss"):

            loss = tf.nn.ctc_loss(labels=labels,
                                  logits=logits,
                                  label_length=None, logit_length=seq_len, blank_index=-1)  # logits_time_major=False,

            loss = tf.reduce_mean(loss)


        return loss

    def compute_metrics(self, labels, logits, seq_len):

        # decoded, log_prob = \
        #     tf.nn.ctc_beam_search_decoder(inputs=logits,
        #                                   sequence_length=seq_len)

        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=False)  # this is faster

        # print(decoded[0], log_prob)
        dense_decoded = tf.sparse.to_dense(decoded[0], default_value=-1)
        # print(dense_decoded)

        # ####Evaluating
        # self.logitsMaxTest = tf.slice(tf.argmax(self.logits, 2), [0, 0], [self.seq_len[0], 1])
        label_error_rate = tf.reduce_mean(
            input_tensor=tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

        return dense_decoded, label_error_rate

    def random_batch(self, X, y, batch_size=32):
        idx = np.random.randint(len(X), size=batch_size)
        X_batch = []
        y_batch = []

        for i in idx:
            X_batch.append(X[i])
            y_batch.append(y[i])

        X_batch = np.asarray(X_batch)
        y_batch = np.asarray(y_batch)
        
        return X_batch, y_batch

    @tf.function
    def sparse_tuple_from_label(self, sequences, dtype=np.int32):
            """Create a sparse representention of x.
            Args:
                sequences: a list of lists of type dtype where each element is a sequence
            Returns:
                A tuple with (indices, values, shape)
            """
            indices = []
            values = []

            for n, seq in enumerate(sequences):
                indices.extend(zip([n] * len(seq), range(len(seq))))
                values.extend(seq)

            indices = np.asarray(indices, dtype=np.int64)
            values = np.asarray(values, dtype=dtype)
            shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

            return tf.sparse.SparseTensor(indices, values, shape)  # (indices, values, shape)

    def training(self, X_train, X_val, y_train, y_val):
        n_epochs = 1000
        batch_size=32
        n_steps=len(X_train)//batch_size
        patience = 30
        min_delta = 0.001

        best_val_acc = 0
        loc_patience = 0

        # for loop iterate over epochs
        for epoch in range(n_epochs):

            print("Epoch {}/{}".format(epoch, n_epochs))
            if loc_patience >= patience:
                print("Early Stopping!!!")
                break

            start_time = time.time()
            train_loss = []
            train_acc = []
            val_loss = []
            val_acc = []

            # for loop iterate over batches
            for step in range(1, n_steps + 1):
                
                X_batch, y_batch_ori=self.random_batch(X_train, y_train)
                indices = []
                values = []

                for n, seq in enumerate(y_batch_ori):
                    indices.extend(zip([n] * len(seq), range(len(seq))))
                    values.extend(seq)

                indices = np.asarray(indices, dtype=np.int64)
                values = np.asarray(values, dtype=np.int32)
                shape = np.asarray([len(y_batch_ori), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

                y_batch = tf.sparse.SparseTensor(indices, values, shape)
                batch_seq_len = np.asarray([(self.char_nums * self.img_width) // 16] * self.batch_size, dtype=np.int32)

                with tf.GradientTape() as tape:
                    y_pred=self.model(X_batch, training=True)
                    loss = self.compute_loss(y_batch, y_pred, batch_seq_len)
                gradients=tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

                dense_decoded, label_error_rate = self.compute_metrics(y_batch, y_pred, batch_seq_len)
                accuracy = self.accuracy_calculation(y_batch_ori, dense_decoded,
                                             ignore_value=-1)

                train_loss.append(loss)
                train_acc.append(accuracy)

                # Read out training results
                # now = datetime.datetime.now()
                # log = "{}/{} {}:{}:{} global_step {}, " \
                #     "accuracy = {:.3f},train_loss = {:.3f}, " \
                #     "label_error_rate = {:.3f}, train using time = {:.3f}"

                # print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                #                 self.optimizer.iterations.numpy(), accuracy, loss,
                #                 label_error_rate, time.time() - start_time))
                # if self.optimizer.iterations.numpy() % 10 == 0:
                #     self.checkpoint.save(self.checkpoint_prefix)

                # Run a validation loop at the end of each epoch

            for valbatch in range(1+ n_steps +1):
                X_batchVal, y_batchVal_ori = self.random_batch(X_val, y_val)
                indices = []
                values = []

                for n, seq in enumerate(y_batchVal_ori):
                    indices.extend(zip([n] * len(seq), range(len(seq))))
                    values.extend(seq)

                indices = np.asarray(indices, dtype=np.int64)
                values = np.asarray(values, dtype=np.int32)
                shape = np.asarray([len(y_batchVal_ori), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

                y_batchVal = tf.sparse.SparseTensor(indices, values, shape)
                batch_seq_len = np.asarray([(self.char_nums * self.img_width) // 16] * self.batch_size, dtype=np.int32)

                val_logits = self.model(X_batchVal)
                # Update val metrics
                loss = self.compute_loss(y_batchVal, val_logits, batch_seq_len)
                dense_decoded, label_error_rate = self.compute_metrics(y_batchVal, val_logits, batch_seq_len)
                accuracy = self.accuracy_calculation(y_batchVal_ori, dense_decoded,
                                             ignore_value=-1)

                val_loss.append(loss)
                val_acc.append(accuracy)
                
                # log = "{}/{} {}:{}:{} " \
                #     "accuracy = {:.3f},val_loss = {:.3f}, " \
                #     "label_error_rate = {:.3f}"
                # print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                #          accuracy, loss, label_error_rate))

            log = "train using time = {:.3f}, " \
                "train_accuracy = {:.3f}, train_loss = {:.3f}, " \
                "val_accuracy = {:.3f}, val_loss = {:.3f}"


            print(log.format(time.time() - start_time,
                            np.mean(train_acc), np.mean(train_loss),
                            np.mean(val_acc), np.mean(val_loss)))
            if self.optimizer.iterations.numpy() % 10 == 0:
                self.checkpoint.save(self.checkpoint_prefix)

            # Code for manual Early Stopping:
            if (np.mean(val_acc) - best_val_acc) >= min_delta:
                 best_val_acc = np.mean(val_acc)
                 loc_patience = 0

            else:
                loc_patience += 1

        self.save()


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

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
        onehot = np.zeros(num_classes)             
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
        idx.append(id)
        images_gray.append(img_gray.astype(np.float32))
        images.append(im.astype(np.float32))
        labels.append([label])
        label_one_hot.append(onehot)
        
            
        i += 1
    # sparced_labels = sparse_tuple_from_label(labels)
        
    return idx, images, images_gray, labels, label_one_hot


def train():
    ##Further split train data to training set and validation set
    train_idx, train_image, train_image_gray, train_label, train_label_one_hot = convert_from_tfrecords('datasets/tfrecords/train_'+ str(width_re) + '_' + str(height_re) +'.tfrecords')
    print(len(train_idx))

    x_data = np.array(train_image_gray).reshape(train_tfrecord_length, width_re, height_re, 1)
    y_data = train_label
    #y_data = np.array(train_label_one_hot)

    X_train, X_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.2, random_state=1)

    model = ocr_network()
    train = model.training(X_train, X_val, y_train, y_val)
    

if __name__ == '__main__':

    f = open('training data dic.txt', encoding='utf_8_sig')
    words = f.read().splitlines()

    train()



