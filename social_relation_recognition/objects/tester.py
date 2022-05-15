import os
import sys
import time
from collections import OrderedDict

import tensorflow as tf
import yaml
from tools.dataset import (get_dataset, get_inputs, parse_dataset,
                           shuffle_dataset)
from tools.metrics import multi_scores
from tools.utils import get_feature_probs, get_probs, get_time

####################################################################################################################################
##                                                         Global Dictionaries                                                    ##
####################################################################################################################################

# Class split
class2split = {
    3: 'coarse',
    5: 'domain',
    6: 'fine',
    16: 'relation'
}

# Class dataset
class2dataset = {
    3: 'PISC',
    5: 'PIPA',
    6: 'PISC',
    16: 'PIPA'
}

# Features formated
feature2Feature = {
    'bodies_age': 'Age',
    'bodies_gender': 'Gender',
    'bodies_clothing': 'Clothing',
    'bodies_activity': 'Individual Activity',
    'context_activity': 'Contextual Activity',
    'context_emotion': 'Contextual Emotion',
    'first_glance': 'First Glance',
                    'objects_attention': 'Objects'
}

####################################################################################################################################
##                                                              Tester                                                            ##
####################################################################################################################################


class Tester(object):

    def __init__(self, args):

        # Configurations
        self.name = args.name
        self.type = args.type
        self.metric = args.metric

        # Information
        self.time_start = 0.

        # Paths
        self.path_configs = args.path_configs
        self.path_features = args.path_features
        self.path_models = args.path_models

        self.sanity_check()
        self.get_configs()
        self.show_configs()
        self.set_paths()

    def sanity_check(self):

        print('>> [SANITY CHECK]')

        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        """
            Level | Level for Humans | Level Description
           -------|------------------|------------------------------------
            0     | DEBUG            | [Default] Print all messages
            1     | INFO             | Filter out INFO messages
            2     | WARNING          | Filter out INFO & WARNING messages
            3     | ERROR            | Filter out all messages
        """

        print(">>   Environment Variables: ")
        print(">>     - HDF5_USE_FILE_LOCKING:",
              os.environ['HDF5_USE_FILE_LOCKING'])
        print(">>     - TF_CPP_MIN_LOG_LEVEL:",
              os.environ['TF_CPP_MIN_LOG_LEVEL'])

        print(">>   Tensorflow Version:", tf.version.VERSION)

        gpus = tf.config.experimental.list_physical_devices('GPU')

        if(gpus):
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(">>   Devices: ")
                print(">>     -", len(gpus), "Physical GPU(s)")
                print(">>     -", len(logical_gpus), "Logical GPU(s)")
            except RuntimeError as e:
                print(e)

        else:
            print(">> [ERROR] No GPU(s) device(s) available")
            sys.exit()

    def get_configs(self):

        configs_path = os.path.join(
            self.path_configs, self.name, "logs", "configs_" + self.name + ".yaml")

        with open(configs_path) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        self.configs = []

        temp = 'Name: %s' % self.name
        self.configs.append(temp)

        temp = "Metric: %s %s" % (self.type, self.metric)
        self.configs.append(temp)

        self.architecture = data["Architecture"]
        temp = 'Architecture: %s' % (self.architecture)
        self.configs.append(temp)

        self.num_classes = data["Classes"]
        temp = 'Classes: %d' % self.num_classes
        self.configs.append(temp)

        self.features = data["Features"]
        temp = 'Features:'
        self.configs.append(temp)

        self.num_features = len(self.features)

        for feature in self.features:
            temp = '  - %s' % (feature)
            self.configs.append(temp)

        self.num_steps = data["Time Steps"]
        temp = 'Time Steps: %d' % self.num_steps
        self.configs.append(temp)

        self.size_hidden_state = data["Hidden Size"]
        temp = 'Hidden Size: %d' % self.size_hidden_state
        self.configs.append(temp)

        self.loss_type = data["Loss"]
        temp = 'Loss: %s' % (self.loss_type)
        self.configs.append(temp)

        self.loss_reduction = data["Loss Reduction"]
        temp = 'Loss Reduction: %s' % (self.loss_reduction)
        self.configs.append(temp)

        self.optimizer_type = data["Optimizer"]
        temp = 'Optimizer: %s' % (self.optimizer_type)
        self.configs.append(temp)

        self.rate_learning = data["Learning Rate"]
        temp = 'Learning Rate: %f' % self.rate_learning
        self.configs.append(temp)

        self.rate_weight = data["Weight Regularization"]
        temp = 'Weight Regularization: %f' % self.rate_weight
        self.configs.append(temp)

        self.rate_decay = data["Decay Rate"]
        temp = 'Decay Rate: %f' % self.rate_decay
        self.configs.append(temp)

        self.rate_dropout = data["Dropout Rate"]
        temp = 'Dropout Rate: %f' % self.rate_dropout
        self.configs.append(temp)

        self.class_balance = data["Class Balance"]
        temp = 'Class Balance: %s' % (self.class_balance)
        self.configs.append(temp)

        self.norm = data["Node Normalization"]
        temp = 'Node Normalization: %s' % (str(self.norm).lower())
        self.configs.append(temp)

        self.seed = data["Seed"]
        temp = 'Seed: %s' % (self.seed)
        self.configs.append(temp)

        self.eager = data["Mode"]
        if(self.eager):
            temp = 'Mode: eager'
        else:
            temp = 'Mode: graph'

        self.configs.append(temp)

    def show_configs(self):

        print('>> [CONFIGURATIONS]')

        for config in self.configs:
            print('>>   %s' % (config))

    def set_paths(self):

        self.path_features = os.path.join(
            self.path_features, class2dataset[self.num_classes], 'features', 'tfrecords', class2split[self.num_classes])
        self.path_models = os.path.join(
            self.path_models, class2dataset[self.num_classes], class2split[self.num_classes], self.name)

    def load_model(self):

        load_path = os.path.join(
            self.path_models, self.type + "_best_" + self.metric + ".h5")

        print(">>   Loading model:")
        print(">>     " + load_path)

        assert os.path.exists(load_path), \
            ">> [ERROR] Incorrect model path"

        self.model = tf.keras.models.load_model(load_path)

    def test(self):

        print(">> [TEST]")
        print(">>   Preparing dataset...")

        num_samples, dataset = get_dataset(self.path_features, "test")
        dataset = parse_dataset(dataset)

        print(">>   ... done!")
        print(">>   Initializing network ...")

        self.load_model()

        print(">>   ... done!")
        print(">>   Starting the test...")

        self.time_start = time.time()

        all_labels = []
        probs_max = []
        probs_mean = []
        probs_vote = []
        probs_features = {}

        for feature in self.features:
            probs_features[feature] = []

        step = 0
        porcent = 0

        for sample in dataset:

            if(int(step % (num_samples * 0.1)) == 0):
                print(">>     {0:.0%}".format(porcent))
                porcent += 0.1

            all_labels.append(sample['label'].numpy())

            inputs, labels = get_inputs(
                sample, self.features, self.num_classes, normalize=self.norm)

            out_probs = self.model.predict_on_batch(inputs)

            get_probs(out_probs, probs_max, probs_mean, probs_vote)
            get_feature_probs(self.features, out_probs, probs_features, sample)

            step += 1

        print(">>     {0:.0%}".format(porcent))

        assert num_samples == step, \
            ">> [ERROR] Incorrect number of samples on validation"

        assert len(probs_max) == num_samples, \
            ">> [ERROR] Problem on getting max probabilities"
        assert len(probs_mean) == num_samples, \
            ">> [ERROR] Problem on getting mean probabilities"
        assert len(probs_vote) == num_samples, \
            ">> [ERROR] Problem on getting vote probabilities"

        for feature in probs_features:
            assert len(probs_features[feature]) == num_samples, \
                ">> [ERROR] Problem on getting features probabilities"

        dict_probs = OrderedDict()

        for feature in probs_features:
            dict_probs[feature2Feature[feature]] = probs_features[feature]

        dict_probs['Max'] = probs_max
        dict_probs['Mean'] = probs_mean
        dict_probs['Vote'] = probs_vote

        self.process_probs(dict_probs, all_labels, 'Test')

        print(">>   [Time]")
        print(">>     Total: " + get_time(time.time() - self.time_start))
        print(">>   ... test finished!")

    def process_probs(self, dict, labels, mode):

        print('>>   [Metrics: ' + mode + ']')

        for key in dict:

            scores = multi_scores(dict[key], labels)

            print('>>     - ' + key + '\n'
                  '>>       - Loss: {0}\n'
                  '>>       - Accuracy: {1}\n'
                  '>>       - Balanced Acc: {2}\n'
                  '>>       - Mean AP: {3}\n'
                  '>>       - Class Recall: {4}\n'
                  '>>       - Class Precision: {5}\n'
                  '>>       - Class Average Precision: {6}'.format(
                      scores['log_loss'],
                      scores['accuracy'],
                      scores['balanced_accuracy'],
                      scores['mean_average_precision'],
                      scores['class_recall'],
                      scores['class_precision'],
                      scores['class_average_precision']
                  ))
