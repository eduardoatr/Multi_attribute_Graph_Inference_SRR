import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tools.dict import get_data, get_paths

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Dictionaries & Lists
partitions_list = [
    'train',
    'eval',
    'test'
]

input2feature = {
    'input_1': 'first_glance',
    'input_2': 'context_activity',
    'input_3': 'context_emotion',
    'input_4': 'bodies_age',
    'input_5': 'bodies_gender',
    'input_6': 'bodies_clothing',
    'input_7': 'bodies_activity',
    'input_objects': 'objects_attention'
}

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

input2size = {
    'input_1': feature2size[input2feature['input_1']],
    'input_2': feature2size[input2feature['input_2']],
    'input_3': feature2size[input2feature['input_3']],
    'input_4': feature2size[input2feature['input_4']],
    'input_5': feature2size[input2feature['input_5']],
    'input_6': feature2size[input2feature['input_6']],
    'input_7': feature2size[input2feature['input_7']],
    'input_objects': feature2size[input2feature['input_objects']]
}

sample2feature = {
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

type2class = {
    'coarse': 3,
    'domain': 5,
    'fine': 6,
    'relation': 16
}


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(data_dict):

    features = {}

    for key in data_dict:

        if(key != 'label'):
            feature_name = input2feature[key]
            feature_list = data_dict[key]
            feature_flatten = (np.asarray(feature_list).flatten()).tolist()

            features[feature_name] = _float_array_feature(feature_flatten)

        else:
            features['label'] = _int64_feature(data_dict[key])

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features))

    return example_proto.SerializeToString()


def convert(partition, args):

    dict_paths = get_paths(args.type, partition, args.relations_dir,
                           args.features_dir, input2feature, type2class)
    num_samples = len(dict_paths['label'])

    print(">> Number of samples: " + str(num_samples))

    filename = os.path.join(args.save_dir, partition + '.tfrecords')

    step = 0
    porcent = 0

    with tf.io.TFRecordWriter(filename) as writer:
        while step < num_samples:

            if(int(step % (num_samples * 0.1)) == 0):
                print(">>   {0:.0%}".format(porcent))
                porcent += 0.1

            dict_data = get_data(dict_paths, step, input2size)

            example = serialize_example(dict_data)

            writer.write(example)

            step += 1

        print(">>   {0:.0%}".format(porcent))


def _parse_feature_function(example_proto):
    return tf.io.parse_single_example(example_proto, sample2feature)


def test(partition, args):

    filename = os.path.join(args.save_dir, partition + '.tfrecords')

    raw_dataset = tf.data.TFRecordDataset(filenames=[filename])

    parsed_dataset = raw_dataset.map(_parse_feature_function)

    dict_paths = get_paths(args.type, partition, args.relations_dir,
                           args.features_dir, input2feature, type2class)

    num_samples = len(dict_paths['label'])

    print(">> Number of samples: " + str(num_samples))

    step = 0
    porcent = 0

    for sample in parsed_dataset:

        dict_data = get_data(dict_paths, step, input2size)

        for key in dict_data:
            if(key != 'label'):

                feature_name = input2feature[key]
                feature_old = np.asarray(dict_data[key])
                feature_new = sample[feature_name].numpy()

                assert np.array_equal(feature_old, feature_new), \
                    ">> [ERROR] Features " + feature_name + " are not the same"

            else:

                feature = dict_data[key]
                feature_new = sample[key].numpy()

                assert feature == feature_new, \
                    ">> [ERROR] Labels are not the same"

        if(int(step % (num_samples * 0.1)) == 0):
            print(">>   {0:.0%}".format(porcent))
            porcent += 0.1

        step += 1

    print(">>   {0:.0%}".format(porcent))

    assert num_samples == step, \
        ">> [ERROR] Incorrect number o samples"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert features')
    parser.add_argument('--relations_dir', type=str, metavar='DIR',
                        help='path to the input data')
    parser.add_argument('--features_dir', type=str, metavar='DIR',
                        help='path to the features')
    parser.add_argument('--save_dir', type=str, metavar='DIR',
                        help='path to save the converted features')
    parser.add_argument('--type', choices=['fine', 'coarse', 'relation', 'domain'],
                        type=str, help='type of dataset')
    parser.add_argument('--mode', choices=['convert', 'test', 'both'],
                        type=str, help='execution mode')

    args = parser.parse_args()

    for partition in partitions_list:

        if(args.mode == 'convert' or args.mode == 'both'):
            print(">> Starting the convertion for " +
                  partition + " features...")
            convert(partition, args)
            print(">> ...done!")

        if(args.mode == 'test' or args.mode == 'both'):
            print(">> Testing " + partition + " features...")
            test(partition, args)
            print(">> ...done!")
