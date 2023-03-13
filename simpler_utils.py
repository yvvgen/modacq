import numpy as np
import h5py
import os
import pandas as pd
import csv
import tensorflow as tf

unique_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

with open('item_list_train.csv', 'r') as f:
    reader = csv.reader(f)
    train_files = list(reader)[0]


def getIndex(meta_path, file_list):
    meta_file = pd.read_csv(meta_path, sep='\t')
    name_files = meta_file['filename'].to_list()
    index_file = [name_files.index('audio/'+file) for file in file_list]
    return index_file

def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / len(labels))
    return labels

def create_one_hot_encoding(word, unique_words):
    """Creates an one-hot encoding of the `word` word, based on the\
    list of unique words `unique_words`.
    """
    to_return = np.zeros((len(unique_words)))
    to_return[unique_words.index(word)] = 1
    return to_return

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

def get_data(hdf5_path, index_file):
    with h5py.File(hdf5_path, 'r') as hf:
        features = int16_to_float32(hf['features'][index_file])
        labels = [f.decode() for f in hf['scene_label'][index_file]]
        acquisition_modality = [f.decode() for f in hf['source_label'][index_file]]
        audio_name = [f.decode() for f in hf['filename'][index_file]]
    return features, labels, audio_name, acquisition_modality

def load_data(features_path, meta_path, split, unique_words=unique_labels, files=train_files):
    Y = []
    index_files = getIndex(meta_path, files)
    # Get files from h5
    features, labels, audio_name, modac = get_data(features_path, index_files)
    for lab in labels:
        if split == 'Train':
            Y.append(smooth_labels(create_one_hot_encoding(lab, unique_words)))
        else:
            Y.append(create_one_hot_encoding(lab, unique_words))

    data_size = {
        'data': features[0].shape[0],
        'time': features[0].shape[1],
    }
    
    features = tf.expand_dims(features, axis=-1)
    Y = tf.expand_dims(Y, axis=-1)
    
    return features, Y, labels, modac, data_size

