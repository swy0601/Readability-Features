import math
import os
import random
import re
import cv2
import numpy as np
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


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
embedding_dim = 100
# store all data
data_set = {}

# store file name
file_name = []

# store structure information
data_structure = {}

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


texts = {}

token_index = {}

sequences = {}


def process_texture_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(texture_dir, label_type)
        for f_name in os.listdir(dir_name):
            if f_name[-4:] == ".txt":
                f = open(os.path.join(dir_name, f_name), errors='ignore')
                content = f.read()
                content = re.sub(pattern, lambda x: " " + x.group(0), content)
                content = re.sub(pattern1, lambda x: " " + x.group(0) + " ", content)
                content = re.sub(pattern2, lambda x: " " + x.group(0) + " ", content)
                content = re.sub(pattern3, lambda x: " ", content)
                # content = re.sub(r'[{}]+'.format(string.punctuation), ' ', content)
                # content = content.strip()
                texts[f_name.split('.')[0]] = content
    token_index = {}
    for sample in texts.values():
        for word in sample.split():
            if word not in token_index:
                if len(word) > 1:
                    token_index[word] = len(token_index) + 1
                else:
                    if not word.isalpha():
                        token_index[word] = len(token_index) + 1
        max_length = 10
        results = np.zeros(shape=(len(texts), max_length, max(token_index.values()) + 1))
        for i, sample in enumerate(texts):
            for j, word in list(enumerate(sample.split()))[:max_length]:
                index = token_index.get(word)
                results[i, j, index] = 1
    print('Found %s unique tokens.' % len(token_index))

    for sample in texts.keys():
        sequence = []
        num = 0
        for word in texts[sample].split():
            if word in token_index.keys() and num < max_len and token_index[word] < max_words:
                sequence.append(token_index[word])
                num += 1
        sequences[sample] = sequence


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
        token.append(sequences[item])


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_1 = true_positives / (possible_positives + K.epsilon())
    return recall_1


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_1 = true_positives / (predicted_positives + K.epsilon())
    return precision_1


def f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    score = 2 * (pre * rec) / (pre + rec)
    return score


def create_NetTGlove():
    texture_input = keras.Input(shape=(100,), name='texture')
    texture_embedded = keras.layers.Embedding(max_words, embedding_dim)(texture_input)
    texture_conv1 = keras.layers.Conv1D(32, 5, activation='relu')(texture_embedded)
    texture_pool1 = keras.layers.MaxPool1D(3)(texture_conv1)
    texture_conv2 = keras.layers.Conv1D(32, 5, activation='relu')(texture_pool1)

    texture_gru = keras.layers.Bidirectional(keras.layers.LSTM(32))(texture_conv2)
    dense1 = keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(texture_gru)
    drop = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')(drop)
    dense3 = keras.layers.Dense(1, activation='sigmoid')(dense2)
    model = keras.Model(texture_input, dense3)
    rms = keras.optimizers.RMSprop(lr=0.0015)
    # model.summary()
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', f1, 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])
    return model


embeddings_index = {}
embedding_matrix = np.zeros((max_words, embedding_dim))


def glove():
    glove_dir = '../Relevant Library/glove 2'
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), errors='ignore')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    for word, i in token_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


if __name__ == '__main__':
    preprocess_structure_data()
    process_texture_data()
    preprocess_picture_data()
    random_dataSet()
    glove()
    # format the data
    label = np.asarray(label)
    structure = np.asarray(structure)
    image = np.asarray(image)
    token = pad_sequences(token, maxlen=max_len)
    segment = np.asarray(segment)

    print('Shape of structure data tensor:', structure.shape)
    print('Shape of image data tensor:', image.shape)
    print('Shape of token tensor:', token.shape)
    print('Shape of segment tensor:', segment.shape)
    print('Shape of label tensor:', label.shape)

    train_structure = structure[21:]
    test_structure = structure[:21]

    train_image = image[21:]
    test_image = image[:21]

    train_token = token[21:]
    test_token = token[:21]

    train_segment = segment[21:]
    test_segment = segment[0:21]

    train_label = label[21:]
    test_label = label[:21]

    print('Shape of train structure data tensor:', train_structure.shape)
    print('Shape of train image data tensor:', train_image.shape)
    print('Shape of train token data tensor:', train_token.shape)
    print('Shape of train segment data tensor:', train_segment.shape)
    print('Shape of train label data tensor:', train_label.shape)
    print('Shape of test structure data tensor:', test_structure.shape)
    print('Shape of test image data tensor:', test_image.shape)
    print('Shape of test token data tensor:', test_token.shape)
    print('Shape of test segment data tensor:', test_segment.shape)
    print('Shape of test label data tensor:', test_label.shape)

    k_fold = 10
    num_sample = len(train_label) // k_fold
    train_t_acc = []
    history_t_list = []

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

        T_model = create_NetTGlove()

        filepath_t = "../Experimental output/BEST1.hdf5"

        checkpoint_t = ModelCheckpoint(filepath_t, monitor='val_acc', verbose=1, save_best_only=True,
                                       model='max')
        callbacks_t_list = [checkpoint_t]

        history_t = T_model.fit(x_train_token, y_train,
                                epochs=30, batch_size=42, callbacks=callbacks_t_list, verbose=0,
                                validation_data=(x_val_token, y_val))

        history_t_list.append(history_t)

    # data analyze
    best_val_f1_t = []
    best_val_auc_t = []
    best_val_mcc_t = []

    epoch_time_t = 1
    for history_item in history_t_list:
        MCC_T = []
        F1_T = []
        history_dict = history_item.history
        val_acc_values = history_dict['val_acc']
        val_recall_value = history_dict['val_recall']
        val_precision_value = history_dict['val_precision']
        val_f1_value = history_dict['val_f1']
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
                MCC_T.append(result_mcc)
                result_precision = tp / (tp + fp)
                result_recall = tp / (tp + fn)
                result_f1 = 2 * result_precision * result_recall / (result_precision + result_recall)
                F1_T.append(result_f1)
        train_t_acc.append(np.max(val_acc_values))
        best_val_f1_t.append(np.max(F1_T))
        best_val_auc_t.append(np.max(val_auc_value))
        best_val_mcc_t.append(np.max(MCC_T))
        print('Processing fold #', epoch_time_t)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        print('best f1 score is #', np.max(F1_T))
        print('best auc score is #', np.max(val_auc_value))
        print('best mcc score is #', np.max(MCC_T))
        print()
        print()
        epoch_time_t = epoch_time_t + 1

    print('Average Glove model acc score', np.mean(train_t_acc))
    print('Average Glove model f1 score', np.mean(best_val_f1_t))
    print('Average Glove model auc score', np.mean(best_val_auc_t))
    print('Average Glove model mcc score', np.mean(best_val_mcc_t))
    print()
