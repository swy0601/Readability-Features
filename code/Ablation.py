import math
import os
import random
import re

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.util import tf_inspect
from transformers import BertTokenizer


# Define the basic bert class
class BertConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.pop('vocab_size', 30000)
        self.type_vocab_size = kwargs.pop('type_vocab_size', 300)
        self.hidden_size = kwargs.pop('hidden_size', 768)
        self.num_hidden_layers = kwargs.pop('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 12)
        self.intermediate_size = kwargs.pop('intermediate_size', 3072)
        self.hidden_activation = kwargs.pop('hidden_activation', 'gelu')
        self.hidden_dropout_rate = kwargs.pop('hidden_dropout_rate', 0.1)
        self.attention_dropout_rate = kwargs.pop('attention_dropout_rate', 0.1)
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 200)
        self.max_sequence_length = kwargs.pop('max_sequence_length', 200)


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(name='BertEmbedding')
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.token_embedding = self.add_weight('weight', shape=[self.vocab_size, self.hidden_size],
                                               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.type_vocab_size = config.type_vocab_size

        self.position_embedding = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='position_embedding'
        )
        self.token_type_embedding = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='token_type_embedding'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)

    def build(self, input_shape):
        with tf.name_scope('bert_embeddings'):
            super().build(input_shape)

    def call(self, inputs, training=False, mode='embedding'):
        # used for masked lm
        if mode == 'linear':
            return tf.matmul(inputs, self.token_embedding, transpose_b=True)

        input_ids, token_type_ids = inputs
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        position_ids = tf.range(input_ids.shape[1], dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids.shape.as_list(), 0)

        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).

        Returns:
            Python dictionary.
        """
        all_args = tf_inspect.getfullargspec(self.__init__).args
        config = {
            'name': self.name,
            'trainable': self.trainable,
        }
        if hasattr(self, '_batch_input_shape'):
            config['batch_input_shape'] = self._batch_input_shape
        config['dtype'] = policy.serialize(self._dtype_policy)
        if hasattr(self, 'dynamic'):
            # Only include `dynamic` in the `config` if it is `True`
            if self.dynamic:
                config['dynamic'] = self.dynamic
            elif 'dynamic' in all_args:
                all_args.remove('dynamic')
        expected_args = config.keys()
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]
        # Check that either the only argument in the `__init__` is  `self`,
        # or that `get_config` has been overridden:
        if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
            raise NotImplementedError('Layer %s has arguments in `__init__` and '
                                      'therefore must override `get_config`.' %
                                      self.__class__.__name__)
        return config


# The following part is about defining relevant data path
structure_dir = '../Dataset/Processed Dataset/Structure'
texture_dir = '../Dataset/Processed Dataset/Texture'
picture_dir = '../Dataset/Processed Dataset/Image'

# Use for texture data preprocessing
pattern = "[A-Z]"
pattern1 = '["\\[\\]\\\\]'
pattern2 = "[*.+!$#&,;{}()':=/<>%-]"
pattern3 = '[_]'

# Define basic parameters
max_len = 100
training_samples = 147
validation_samples = 63
max_words = 1000
epoch_num = 20

# store all data
data_set = {}

# store file name
file_name = []

# store structure information
data_structure = {}

# store texture information
data_texture = {}

# store token, position and segment information
data_token = {}
data_position = {}
data_segment = {}
# dic_content = {}

# store the content of each text
string_content = {}

# store picture information
data_picture = {}

# store content of each picture
data_image = []

# 实验部分  --  随机打乱数据
all_data = []
train_data = []
test_data = []

structure = []
image = []
label = []
token = []
segment = []

tokenizer_path = '../Relevant Library/cased_L-12_H-768_A-12/cased_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
print('Successfully load the BertTokenizer')


def preprocess_structure_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(structure_dir, label_type)
        for f_name in os.listdir(dir_name):
            f = open(os.path.join(dir_name, f_name), errors='ignore')
            lines = []
            if not f_name.startswith('.'):
                file_name.append(f_name.split('.')[0])
                for line in f:
                    line = line.strip(',\n')
                    info = line.split(',')
                    info_int = []
                    count = 0
                    for item in info:
                        if count < 305:
                            info_int.append(int(item))
                            count += 1
                    info_int = np.asarray(info_int)
                    lines.append(info_int)
                f.close()
                lines = np.asarray(lines)
                if label_type == 'Readable':
                    data_set[f_name.split('.')[0]] = 0
                else:
                    data_set[f_name.split('.')[0]] = 1
                data_structure[f_name.split('.')[0]] = lines


def process_texture_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(texture_dir, label_type)
        for f_name in os.listdir(dir_name):
            if f_name[-4:] == ".txt":
                list_content = []
                list_position = []
                list_segment = []
                s = ''
                segment_id = 0
                position_id = 0
                count = 0
                f = open(os.path.join(dir_name, f_name), errors='ignore')
                for content in f:
                    content = re.sub(r"([a-z]+)([A-Z]+)", r"\1 \2", content)
                    content = re.sub(pattern1, lambda x: " " + x.group(0) + " ", content)
                    content = re.sub(pattern2, lambda x: " " + x.group(0) + " ", content)
                    content = re.sub(pattern3, lambda x: " ", content)
                    list_value = content.split()
                    for item in list_value:
                        if len(item) > 1 or not item.isalpha():
                            s = s + ' ' + item
                            list_content.append(item)
                            if count < max_len:
                                list_position.append(position_id)
                                position_id += 1
                                list_segment.append(segment_id)
                            count += 1
                    segment_id += 1
                while count < max_len:
                    list_segment.append(segment_id)
                    list_position.append(count)
                    count += 1
                f.close()
                string_content[f_name.split('.')[0]] = s
                data_position[f_name.split('.')[0]] = list_position
                data_segment[f_name.split('.')[0]] = list_segment
                # dic_content[f_name.split('.')[0]] = list_content

        for sample in string_content:
            list_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(string_content[sample]))
            list_token = list_token[:max_len]
            while len(list_token) < max_len:
                list_token.append(0)
            data_token[sample] = list_token


def preprocess_picture_data():
    for label_type in ['readable', 'unreadable']:
        dir_image_name = os.path.join(picture_dir, label_type)
        for f_name in os.listdir(dir_image_name):
            if not f_name.startswith('.'):
                img_data = cv2.imread(os.path.join(dir_image_name, f_name))
                img_data = cv2.resize(img_data, (128, 128))
                result = img_data / 255.0
                data_picture[f_name.split('.')[0]] = result
                data_image.append(result)


def random_dataSet():
    count_id = 0
    while count_id < 210:
        index_id = random.randint(0, len(file_name) - 1)
        all_data.append(file_name[index_id])
        file_name.remove(file_name[index_id])
        count_id += 1
    for item in all_data:
        label.append(data_set[item])
        structure.append(data_structure[item])
        image.append(data_picture[item])
        token.append(data_token[item])
        segment.append(data_segment[item])


def create_NetST():
    structure_input = keras.Input(shape=(50, 305), name='structure')
    structure_reshape = keras.layers.Reshape((50, 305, 1))(structure_input)
    structure_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(structure_reshape)
    structure_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(structure_conv1)
    structure_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(structure_pool1)
    structure_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(structure_conv2)
    structure_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(structure_pool2)
    structure_pool3 = keras.layers.MaxPool2D(pool_size=3, strides=3)(structure_conv3)
    structure_flatten = keras.layers.Flatten()(structure_pool3)

    bert_config = BertConfig(max_sequence_length=max_len)
    token_input = keras.Input(shape=(max_len,), name='token')
    segment_input = keras.Input(shape=(max_len,), name='segment')
    texture_embedded = BertEmbedding(config=bert_config)([token_input, segment_input])
    texture_conv1 = keras.layers.Conv1D(32, 5, activation='relu')(texture_embedded)
    texture_pool1 = keras.layers.MaxPool1D(3)(texture_conv1)
    texture_conv2 = keras.layers.Conv1D(32, 5, activation='relu')(texture_pool1)
    texture_gru = keras.layers.Bidirectional(keras.layers.LSTM(32))(texture_conv2)

    concatenated = keras.layers.concatenate([structure_flatten, texture_gru], axis=-1)

    dense1 = keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(concatenated)
    drop = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(units=32, activation='relu', name='random_detail')(drop)
    dense3 = keras.layers.Dense(1, activation='sigmoid')(dense2)
    model = keras.Model([structure_input, token_input, segment_input], dense3)
    rms = keras.optimizers.RMSprop(lr=0.0015)
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])
    return model


def create_NetSV():
    bert_config = BertConfig(max_sequence_length=max_len)
    token_input = keras.Input(shape=(max_len,), name='token')
    segment_input = keras.Input(shape=(max_len,), name='segment')
    texture_embedded = BertEmbedding(config=bert_config)([token_input, segment_input])
    texture_conv1 = keras.layers.Conv1D(32, 5, activation='relu')(texture_embedded)
    texture_pool1 = keras.layers.MaxPool1D(3)(texture_conv1)
    texture_conv2 = keras.layers.Conv1D(32, 5, activation='relu')(texture_pool1)
    texture_gru = keras.layers.Bidirectional(keras.layers.LSTM(32))(texture_conv2)

    image_input = keras.Input(shape=(128, 128, 3), name='image')
    image_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(image_input)
    image_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv1)
    image_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(image_pool1)
    image_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv2)
    image_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(image_pool2)
    image_pool3 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv3)
    image_flatten = keras.layers.Flatten()(image_pool3)

    concatenated = keras.layers.concatenate([texture_gru, image_flatten], axis=-1)

    dense1 = keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(concatenated)
    drop = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(units=32, activation='relu', name='random_detail')(drop)
    dense3 = keras.layers.Dense(1, activation='sigmoid')(dense2)
    model = keras.Model([token_input, segment_input, image_input], dense3)
    rms = keras.optimizers.RMSprop(lr=0.0015)
    # model.summary()
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])
    return model


def create_NetTV():
    structure_input = keras.Input(shape=(50, 305), name='structure')
    structure_reshape = keras.layers.Reshape((50, 305, 1))(structure_input)
    structure_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(structure_reshape)
    structure_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(structure_conv1)
    structure_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(structure_pool1)
    structure_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(structure_conv2)
    structure_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(structure_pool2)
    structure_pool3 = keras.layers.MaxPool2D(pool_size=3, strides=3)(structure_conv3)
    structure_flatten = keras.layers.Flatten()(structure_pool3)

    image_input = keras.Input(shape=(128, 128, 3), name='image')
    image_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(image_input)
    image_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv1)
    image_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(image_pool1)
    image_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv2)
    image_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(image_pool2)
    image_pool3 = keras.layers.MaxPool2D(pool_size=2, strides=2)(image_conv3)
    image_flatten = keras.layers.Flatten()(image_pool3)

    concatenated = keras.layers.concatenate([structure_flatten, image_flatten], axis=-1)

    dense1 = keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(concatenated)
    drop = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(units=32, activation='relu', name='random_detail')(drop)
    dense3 = keras.layers.Dense(1, activation='sigmoid')(dense2)
    model = keras.Model([structure_input, image_input], dense3)
    rms = keras.optimizers.RMSprop(lr=0.0015)
    # model.summary()
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])
    return model


if __name__ == '__main__':
    preprocess_structure_data()
    process_texture_data()
    preprocess_picture_data()
    random_dataSet()

    # format the data
    label = np.asarray(label)
    structure = np.asarray(structure)
    image = np.asarray(image)
    token = np.asarray(token)
    segment = np.asarray(segment)

    print('Shape of structure data tensor:', structure.shape)
    print('Shape of image data tensor:', image.shape)
    print('Shape of token tensor:', token.shape)
    print('Shape of segment tensor:', segment.shape)
    print('Shape of label tensor:', label.shape)

    train_structure = structure
    # test_structure = structure[:21]

    train_image = image
    # test_image = image[:21]

    train_token = token
    # test_token = token[:21]

    train_segment = segment
    # test_segment = segment[0:21]

    train_label = label
    # test_label = label[:21]

    k_fold = 10
    num_sample = math.ceil(len(train_label) / k_fold)
    # train_pts_acc = []
    train_tv_acc = []
    train_sv_acc = []
    train_st_acc = []

    # history_pts_list = []
    history_tv_list = []
    history_sv_list = []
    history_st_list = []

    for epoch in range(k_fold):
        print('Now is fold {}'.format(epoch))
        x_val_structure = train_structure[epoch * num_sample:(epoch + 1) * num_sample]
        x_val_token = train_token[epoch * num_sample:(epoch + 1) * num_sample]
        x_val_segment = train_segment[epoch * num_sample:(epoch + 1) * num_sample]
        x_val_image = train_image[epoch * num_sample:(epoch + 1) * num_sample]
        y_val = train_label[epoch * num_sample:(epoch + 1) * num_sample]

        x_train_structure_part_1 = train_structure[:epoch * num_sample]
        x_train_structure_part_2 = train_structure[(epoch + 1) * num_sample:]
        x_train_structure = np.concatenate([x_train_structure_part_1, x_train_structure_part_2], axis=0)

        x_train_token_part_1 = train_token[:epoch * num_sample]
        x_train_token_part_2 = train_token[(epoch + 1) * num_sample:]
        x_train_token = np.concatenate([x_train_token_part_1, x_train_token_part_2], axis=0)

        x_train_segment_part_1 = train_segment[:epoch * num_sample]
        x_train_segment_part_2 = train_segment[(epoch + 1) * num_sample:]
        x_train_segment = np.concatenate([x_train_segment_part_1, x_train_segment_part_2], axis=0)

        x_train_image_part_1 = train_image[:epoch * num_sample]
        x_train_image_part_2 = train_image[(epoch + 1) * num_sample:]
        x_train_image = np.concatenate([x_train_image_part_1, x_train_image_part_2], axis=0)

        y_train_part_1 = train_label[:epoch * num_sample]
        y_train_part_2 = train_label[(epoch + 1) * num_sample:]
        y_train = np.concatenate([y_train_part_1, y_train_part_2], axis=0)

        TV_model = create_NetTV()
        SV_model = create_NetSV()
        ST_model = create_NetST()

        filepath_tv = "../Experimental output/TV_BEST.hdf5"
        filepath_sv = "../Experimental output/SV_BEST.hdf5"
        filepath_st = "../Experimental output/ST_BEST.hdf5"

        checkpoint_tv = ModelCheckpoint(filepath_tv, monitor='val_acc', verbose=1, save_best_only=True,
                                        model='max')
        callbacks_tv_list = [checkpoint_tv]

        checkpoint_sv = ModelCheckpoint(filepath_sv, monitor='val_acc', verbose=1, save_best_only=True,
                                        model='max')
        callbacks_sv_list = [checkpoint_sv]

        checkpoint_st = ModelCheckpoint(filepath_st, monitor='val_acc', verbose=1, save_best_only=True,
                                        model='max')
        callbacks_st_list = [checkpoint_st]

        history_tv = TV_model.fit([x_train_structure, x_train_image], y_train,
                                  epochs=epoch_num, batch_size=42, callbacks=callbacks_tv_list, verbose=0,
                                  validation_data=([x_val_structure, x_val_image], y_val))

        history_tv_list.append(history_tv)

        history_tp = SV_model.fit([x_train_token, x_train_segment, x_train_image], y_train,
                                  epochs=epoch_num, batch_size=42, callbacks=callbacks_sv_list, verbose=0,
                                  validation_data=([x_val_token, x_val_segment, x_val_image], y_val))

        history_sv_list.append(history_tp)

        history_st = ST_model.fit([x_train_structure, x_train_token, x_train_segment], y_train,
                                  epochs=epoch_num, batch_size=42, callbacks=callbacks_st_list, verbose=0,
                                  validation_data=([x_val_structure, x_val_token, x_val_segment], y_val))

        history_st_list.append(history_st)

    # data analyze
    best_val_f1_tv = []
    best_val_f1_sv = []
    best_val_f1_st = []

    best_val_auc_tv = []
    best_val_auc_sv = []
    best_val_auc_st = []

    best_val_mcc_tv = []
    best_val_mcc_sv = []
    best_val_mcc_st = []

    epoch_time_tv = 1
    print("The model without texture representation: ")
    for history_item in history_tv_list:
        MCC_TV = []
        F1_TV = []
        history_dict = history_item.history
        val_acc_values = history_dict['val_acc']
        val_recall_value = history_dict['val_recall']
        val_precision_value = history_dict['val_precision']
        val_auc_value = history_dict['val_auc']
        val_false_negatives = history_dict['val_false_negatives']
        val_false_positives = history_dict['val_false_positives']
        val_true_positives = history_dict['val_true_positives']
        val_true_negatives = history_dict['val_true_negatives']
        for i in range(20):
            tp = val_true_positives[i]
            tn = val_true_negatives[i]
            fp = val_false_positives[i]
            fn = val_false_negatives[i]
            if tp > 0 and tn > 0 and fn > 0 and fp > 0:
                result_mcc = (tp * tn - fp * fn) / (math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
                MCC_TV.append(result_mcc)
                result_precision = tp / (tp + fp)
                result_recall = tp / (tp + fn)
                result_f1 = 2 * result_precision * result_recall / (result_precision + result_recall)
                F1_TV.append(result_f1)
        train_tv_acc.append(np.max(val_acc_values))
        best_val_f1_tv.append(np.max(F1_TV))
        best_val_auc_tv.append(np.max(val_auc_value))
        best_val_mcc_tv.append(np.max(MCC_TV))
        print('Processing fold #', epoch_time_tv)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        print('best f1 score is #', np.max(F1_TV))
        print('best auc score is #', np.max(val_auc_value))
        print('best mcc score is #', np.max(MCC_TV))
        print()
        print()
        epoch_time_tv = epoch_time_tv + 1

    epoch_time_sv = 1
    print("The model without structure representation: ")
    for history_item in history_sv_list:
        MCC_SV = []
        F1_SV = []
        history_dict = history_item.history
        val_acc_values = history_dict['val_acc']
        val_recall_value = history_dict['val_recall']
        val_precision_value = history_dict['val_precision']
        val_auc_value = history_dict['val_auc']
        val_false_negatives = history_dict['val_false_negatives']
        val_false_positives = history_dict['val_false_positives']
        val_true_positives = history_dict['val_true_positives']
        val_true_negatives = history_dict['val_true_negatives']
        for i in range(20):
            tp = val_true_positives[i]
            tn = val_true_negatives[i]
            fp = val_false_positives[i]
            fn = val_false_negatives[i]
            if tp > 0 and tn > 0 and fn > 0 and fp > 0:
                result_mcc = (tp * tn - fp * fn) / (math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
                MCC_SV.append(result_mcc)
                result_precision = tp / (tp + fp)
                result_recall = tp / (tp + fn)
                result_f1 = 2 * result_precision * result_recall / (result_precision + result_recall)
                F1_SV.append(result_f1)
        train_sv_acc.append(np.max(val_acc_values))
        best_val_f1_sv.append(np.max(F1_SV))
        best_val_auc_sv.append(np.max(val_auc_value))
        best_val_mcc_sv.append(np.max(MCC_SV))
        print('Processing fold #', epoch_time_sv)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        print('best f1 score is #', np.max(F1_SV))
        print('best auc score is #', np.max(val_auc_value))
        print('best mcc score is #', np.max(MCC_SV))
        print()
        print()
        epoch_time_sv = epoch_time_sv + 1

    epoch_time_st = 1
    print("The model without image representation: ")
    for history_item in history_st_list:
        MCC_ST = []
        F1_ST = []
        history_dict = history_item.history
        val_acc_values = history_dict['val_acc']
        val_recall_value = history_dict['val_recall']
        val_precision_value = history_dict['val_precision']
        val_auc_value = history_dict['val_auc']
        val_false_negatives = history_dict['val_false_negatives']
        val_false_positives = history_dict['val_false_positives']
        val_true_positives = history_dict['val_true_positives']
        val_true_negatives = history_dict['val_true_negatives']
        for i in range(20):
            tp = val_true_positives[i]
            tn = val_true_negatives[i]
            fp = val_false_positives[i]
            fn = val_false_negatives[i]
            if tp > 0 and tn > 0 and fn > 0 and fp > 0:
                result_mcc = (tp * tn - fp * fn) / (math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
                MCC_ST.append(result_mcc)
                result_precision = tp / (tp + fp)
                result_recall = tp / (tp + fn)
                result_f1 = 2 * result_precision * result_recall / (result_precision + result_recall)
                F1_ST.append(result_f1)
        train_st_acc.append(np.max(val_acc_values))
        best_val_f1_st.append(np.max(F1_ST))
        best_val_auc_st.append(np.max(val_auc_value))
        best_val_mcc_st.append(np.max(MCC_ST))
        print('Processing fold #', epoch_time_st)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        print('best f1 score is #', np.max(F1_ST))
        print('best auc score is #', np.max(val_auc_value))
        print('best mcc score is #', np.max(MCC_ST))
        print()
        print()
        epoch_time_st = epoch_time_st + 1


    print('Average S+P model acc score', np.mean(train_tv_acc))
    print('Average S+P model f1 score', np.mean(best_val_f1_tv))
    print('Average S+P model auc score', np.mean(best_val_auc_tv))
    print('Average S+P model mcc score', np.mean(best_val_mcc_tv))
    print()

    print('Average T+P model acc score', np.mean(train_sv_acc))
    print('Average T+P model f1 score', np.mean(best_val_f1_sv))
    print('Average T+P model auc score', np.mean(best_val_auc_sv))
    print('Average T+P model mcc score', np.mean(best_val_mcc_sv))
    print()

    print('Average S+T model acc score', np.mean(train_st_acc))
    print('Average S+T model f1 score', np.mean(best_val_f1_st))
    print('Average S+T model auc score', np.mean(best_val_auc_st))
    print('Average S+T model mcc score', np.mean(best_val_mcc_st))
    print()
