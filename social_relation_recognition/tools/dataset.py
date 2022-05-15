import os

import numpy as np
import tensorflow as tf

# Features size
feature2size = {
    'bodies_age': 4096,
    'bodies_gender': 4096,
    'bodies_clothing': 4096,
    'bodies_activity': 1024,
    'context_activity': 1024,
    'context_emotion': 1024,
    'first_glance': 4096,
    'objects_attention': 2048
}

# Dataset feature type
feature2sample = {
    'bodies_age': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['bodies_age'])]),
    'bodies_gender': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['bodies_gender'])]),
    'bodies_clothing': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['bodies_clothing'])]),
    'bodies_activity': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['bodies_activity'])]),
    'context_activity': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['context_activity'])]),
    'context_emotion': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['context_emotion'])]),
    'first_glance': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['first_glance'])]),
    'objects_attention': tf.io.RaggedFeature(dtype=tf.float32, partitions=[tf.io.RaggedFeature.UniformRowLength(feature2size['objects_attention'])]),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
}


def parse_feature(example_proto):
    return tf.io.parse_single_example(example_proto, feature2sample)


def get_dataset(path_features, split):

    data_path = os.path.join(path_features, split + ".tfrecords")

    assert os.path.isfile(data_path), \
        ">> [ERROR] Incorrect dataset path"

    dataset = tf.data.TFRecordDataset(filenames=[data_path])

    samples = 0

    for sample in dataset:
        samples += 1

    return samples, dataset


def parse_dataset(dataset):
    parsed_dataset = dataset.map(parse_feature)

    return parsed_dataset


def shuffle_dataset(dataset, buffer_size):
    shuffled_dataset = dataset.shuffle(
        buffer_size=buffer_size, reshuffle_each_iteration=False)

    return shuffled_dataset


def get_inputs(sample, features, classes, normalize=True):

    inputs = {}
    count = 0

    for feature in features:

        if(feature == 'bodies_activity'):
            inputs['input_bodies_activity'] = np.asarray(
                sample['bodies_activity'])
            count += 2

        elif(feature == 'bodies_age'):
            inputs['input_bodies_age'] = np.asarray(sample['bodies_age'])
            count += 2

        elif(feature == 'bodies_clothing'):
            inputs['input_bodies_clothing'] = np.asarray(
                sample['bodies_clothing'])
            count += 2

        elif(feature == 'bodies_gender'):
            inputs['input_bodies_gender'] = np.asarray(sample['bodies_gender'])
            count += 2

        elif(feature == 'context_activity'):
            inputs['input_context_activity'] = np.asarray(
                sample['context_activity'])
            count += 1

        elif(feature == 'context_emotion'):
            inputs['input_context_emotion'] = np.asarray(
                sample['context_emotion'])
            count += 1

        elif(feature == 'first_glance'):
            inputs['input_first_glance'] = np.asarray(sample['first_glance'])
            count += 1

        elif(feature == 'objects_attention'):
            inputs['input_objects_attention'] = np.asarray(
                sample['objects_attention'])
            num_objects, _ = inputs['input_objects_attention'].shape
            count += num_objects

    adjacency_matrix = np.ones((count, count)) - np.diag(np.ones(count))

    if(normalize):
        adjacency_matrix = np.multiply(adjacency_matrix, (1./float(count-1)))

    inputs['input_adjacency_matrix'] = adjacency_matrix

    labels = np.repeat(np.asarray(sample['label']), count, axis=0)
    labels = tf.one_hot(labels, classes)

    return inputs, labels
