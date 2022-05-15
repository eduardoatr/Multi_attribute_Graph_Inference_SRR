import os
from random import shuffle

import numpy as np


def get_paths(type, partition, path_relations, path_features, input2feature, type2class):

    dict = {}

    for input in input2feature:
        dict[input] = []

    dict['label'] = []

    for class_num in range(type2class[type]):

        img_list = os.listdir(os.path.join(
            path_relations, partition, str(class_num)))

        for single_img in img_list:

            img_name = single_img.split('.')[0]

            for input in input2feature:
                dict[input].append(os.path.join(path_features, type, partition, str(
                    class_num), img_name, input2feature[input]))

            dict['label'].append(class_num)

    for input in input2feature:
        assert len(dict[input]) == len(dict['label'])

    return dict


def get_data(dict_data, step, input2size):

    dict = {}
    count_feature = {}
    total_len = 0

    for key in dict_data:

        if(key != 'label'):

            dir_list = dict_data[key]
            feature_path = dir_list[step]
            feature_list = os.listdir(feature_path)
            feature_len = len(feature_list)
            dict[key] = []

            total_len += feature_len
            count_feature[key] = feature_len

            if feature_len > 0:
                for i in range(feature_len):

                    feature_tensor = np.load(
                        os.path.join(feature_path, feature_list[i]))
                    dict[key].append(np.reshape(
                        feature_tensor, [input2size[key], ]))

            else:
                dict[key] = np.zeros((1, input2size[key]))

    label = dict_data['label']

    dict['label'] = label[step]

    return dict


def shuffle_dict(self, dict):

    zipped = list(zip(*dict.values()))

    shuffle(zipped)

    shuffled_list = list(zip(*zipped))

    count = 0
    dict_shuffled = {}

    for key in dict:
        dict_shuffled[key] = shuffled_list[count]
        count += 1

    return dict_shuffled
