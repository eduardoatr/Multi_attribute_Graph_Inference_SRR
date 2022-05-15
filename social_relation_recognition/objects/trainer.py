import os
import sys
import time
from collections import OrderedDict

import tensorflow as tf
import tensorflow_addons as tfa
from models.baseline import get_model_baseline
from models.increase import get_model_edge_mlp
from tools.dataset import (get_dataset, get_inputs, parse_dataset,
                           shuffle_dataset)
from tools.metrics import AverageMeter, multi_scores
from tools.utils import get_feature_probs, get_probs, get_time

import objects.logger as l

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

# Class weights
class2weight = {
    3: {0: 0.76, 1: 0.95, 2: 1.},
    6: {0: 0.041, 1: 0.067, 2: 0.337, 3: 0.025, 4: 1., 5: 0.044},
    16: {0: 0.310, 1: 0.223, 2: 2.173, 3: 2.703, 4: 0.033, 5: 0.164, 6: 0.781, 7: 0.199, 8: 0.515, 9: 4.348, 10: 1.205, 11: 10., 12: 0.192, 13: 5.882, 14: 0.116, 15: 0.015}
}

# New class weights
# (1/#samples_class) * (total/#classes)
class2weightnew = {
    3: {0: 0.854055869, 1: 1.065547885, 2: 1.122799741},
    6: {0: 0.727835964, 1: 1.181035662, 2: 5.949313645, 3: 0.443015332, 4: 17.654558003, 5: 0.770789431},
    16: {0: 2.584525602, 1: 1.915318203, 2: 18.653532236, 3: 23.190878355, 4: 0.280963127, 5: 1.411287142, 6: 6.703613281, 7: 1.705890031, 8: 4.423002427, 9: 37.307065329, 10: 10.338102606, 11: 85.80625, 12: 1.650120258, 13: 50.474264353, 14: 0.994279064, 15: 0.125027429}
}

####################################################################################################################################
##                                                             Trainer                                                            ##
####################################################################################################################################


class Trainer(object):

    def __init__(self, args):

        # Configurations
        self.name = args.name
        self.save = args.save
        self.architecture = args.arch
        self.features = args.features
        self.num_classes = args.classes
        self.num_features = len(args.features)
        self.num_epochs = args.epochs
        self.num_steps = args.time
        self.size_hidden_state = args.hidden
        self.rate_learning = args.learning
        self.rate_weight = args.weight
        self.rate_decay = args.decay
        self.rate_dropout = args.dropout
        self.loss_type = args.loss
        self.loss_reduction = args.reduction
        self.optimizer_type = args.optimizer
        self.class_balance = args.balance
        self.seed = args.seed

        if(args.norm == 'true'):
            self.norm = True
        else:
            self.norm = False

        if(args.mode == 'eager'):
            self.eager = True
        else:
            self.eager = False

        # Meters
        self.meter_time = AverageMeter()
        self.meter_accuracy = AverageMeter()
        self.meter_loss = AverageMeter()
        self.meter_loss_focal = AverageMeter()
        self.meter_loss_weighted = AverageMeter()
        self.meter_acc_max = AverageMeter()
        self.meter_acc_mean = AverageMeter()
        self.meter_acc_vote = AverageMeter()
        self.meter_map_max = AverageMeter()
        self.meter_map_mean = AverageMeter()
        self.meter_map_vote = AverageMeter()
        self.meter_loss_max = AverageMeter()
        self.meter_loss_mean = AverageMeter()
        self.meter_loss_vote = AverageMeter()

        # Information
        self.time_start = 0.
        self.time_total = 0.
        self.epoch_best_loss = 0
        self.epoch_best_loss = 0
        self.epoch_best_acc = 0
        self.epoch_best_map = 0
        self.features_loss = {}
        self.features_acc = {}
        self.features_map = {}

        # Paths
        self.path_features = args.path_features
        self.path_save = args.path_save
        self.path_results = args.path_results

        # Run data
        self.logger = None
        self.writer = None
        self.sanity_check()
        self.set_configs()
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

    def set_configs(self):

        self.configs = []

        temp = 'Name: %s' % (self.name)
        self.configs.append(temp)

        temp = 'Save Models:'
        self.configs.append(temp)

        for type in self.save:
            temp = '  - %s' % (type)
            self.configs.append(temp)

        temp = 'Minimun Epochs: %d' % self.num_epochs
        self.configs.append(temp)

        temp = 'Architecture: %s' % (self.architecture)
        self.configs.append(temp)

        temp = 'Classes: %d' % self.num_classes
        self.configs.append(temp)

        temp = 'Features:'
        self.configs.append(temp)

        for feature in self.features:
            temp = '  - %s' % (feature)
            self.configs.append(temp)

        temp = 'Time Steps: %d' % self.num_steps
        self.configs.append(temp)

        temp = 'Hidden Size: %d' % self.size_hidden_state
        self.configs.append(temp)

        temp = 'Loss: %s' % (self.loss_type)
        self.configs.append(temp)

        temp = 'Loss Reduction: %s' % (self.loss_reduction)
        self.configs.append(temp)

        temp = 'Optimizer: %s' % (self.optimizer_type)
        self.configs.append(temp)

        temp = 'Learning Rate: %f' % self.rate_learning
        self.configs.append(temp)

        temp = 'Weight Regularization: %f' % self.rate_weight
        self.configs.append(temp)

        temp = 'Decay Rate: %f' % self.rate_decay
        self.configs.append(temp)

        temp = 'Dropout Rate: %f' % self.rate_dropout
        self.configs.append(temp)

        temp = 'Class Balance: %s' % (self.class_balance)
        self.configs.append(temp)

        temp = 'Node Normalization: %s' % (str(self.norm).lower())
        self.configs.append(temp)

        temp = 'Seed: %s' % (self.seed)
        self.configs.append(temp)

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
        self.path_save = os.path.join(
            self.path_save, class2dataset[self.num_classes], class2split[self.num_classes], self.name)
        self.path_results = os.path.join(
            self.path_results, class2dataset[self.num_classes], class2split[self.num_classes], self.name)

        if(not os.path.exists(self.path_save)):
            os.makedirs(self.path_save)

        if(not os.path.exists(self.path_results)):
            os.makedirs(self.path_results)

    def reset_metrics(self):

        self.meter_accuracy.reset()
        self.meter_loss.reset()
        self.meter_loss_focal.reset()
        self.meter_loss_weighted.reset()

    def update_metrics(self, accuracy, loss):
        self.meter_accuracy.update(accuracy)
        self.meter_loss.update(loss)

    def initiate_logger(self, mode):

        path_logger = os.path.join(self.path_results, 'logs')

        if not os.path.exists(path_logger):
            os.makedirs(path_logger)

        self.logger = l.Logger(self.name, path_logger, mode)

    def initiate_writer(self):

        path_writer = os.path.join(self.path_results, "summaries")

        if not os.path.exists(path_writer):
            os.makedirs(path_writer)

        self.writer = tf.summary.create_file_writer(path_writer)

    def update_summary_train(self, step):

        with self.writer.as_default():

            tf.summary.scalar('Train/Loss', self.meter_loss.val, step=step)
            tf.summary.scalar('Train/Accuracy',
                              self.meter_accuracy.val, step=step)

    def update_summary_eval(self, step):

        with self.writer.as_default():

            tf.summary.scalar(
                'Eval/Loss/Max', self.meter_loss_max.val, step=step)
            tf.summary.scalar('Eval/Loss/Mean',
                              self.meter_loss_mean.val, step=step)
            tf.summary.scalar('Eval/Loss/Vote',
                              self.meter_loss_vote.val, step=step)

            tf.summary.scalar('Eval/Accuracy/Max',
                              self.meter_acc_max.val, step=step)
            tf.summary.scalar('Eval/Accuracy/Mean',
                              self.meter_acc_mean.val, step=step)
            tf.summary.scalar('Eval/Accuracy/Vote',
                              self.meter_acc_vote.val, step=step)

            tf.summary.scalar(
                'Eval/mAP/Max', self.meter_map_max.val, step=step)
            tf.summary.scalar(
                'Eval/mAP/Mean', self.meter_map_mean.val, step=step)
            tf.summary.scalar(
                'Eval/mAP/Vote', self.meter_map_vote.val, step=step)

    def update_summary_features(self, step):

        with self.writer.as_default():

            if('First Glance' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/First_Glance',
                                  self.features_loss['First Glance'], step=step)
            if('First Glance' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/First_Glance',
                                  self.features_acc['First Glance'], step=step)
            if('First Glance' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/First_Glance',
                                  self.features_map['First Glance'], step=step)

            if('Contextual Activity' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/Contextual_Activity',
                                  self.features_loss['Contextual Activity'], step=step)
            if('Contextual Activity' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/Contextual_Activity',
                                  self.features_acc['Contextual Activity'], step=step)
            if('Contextual Activity' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/Contextual_Activity',
                                  self.features_map['Contextual Activity'], step=step)

            if('Contextual Emotion' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/Emotion',
                                  self.features_loss['Contextual Emotion'], step=step)
            if('Contextual Emotion' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/Emotion',
                                  self.features_acc['Contextual Emotion'], step=step)
            if('Contextual Emotion' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/Emotion',
                                  self.features_map['Contextual Emotion'], step=step)

            if('Age' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/Age',
                                  self.features_loss['Age'], step=step)
            if('Age' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/Age',
                                  self.features_acc['Age'], step=step)
            if('Age' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/Age',
                                  self.features_map['Age'], step=step)

            if('Gender' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/Gender',
                                  self.features_loss['Gender'], step=step)
            if('Gender' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/Gender',
                                  self.features_acc['Gender'], step=step)
            if('Gender' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/Gender',
                                  self.features_map['Gender'], step=step)

            if('Clothing' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/Clothing',
                                  self.features_loss['Clothing'], step=step)
            if('Clothing' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/Clothing',
                                  self.features_acc['Clothing'], step=step)
            if('Clothing' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/Clothing',
                                  self.features_map['Clothing'], step=step)

            if('Individual Activity' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/Individual_Activity',
                                  self.features_loss['Individual Activity'], step=step)
            if('Individual Activity' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/Individual_Activity',
                                  self.features_acc['Individual Activity'], step=step)
            if('Individual Activity' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/Individual_Activity',
                                  self.features_map['Individual Activity'], step=step)

            if('Objects' in self.features_loss):
                tf.summary.scalar('Eval/Features/Loss/Objects',
                                  self.features_loss['Objects'], step=step)
            if('Objects' in self.features_acc):
                tf.summary.scalar('Eval/Features/Accuracy/Objects',
                                  self.features_acc['Objects'], step=step)
            if('Objects' in self.features_map):
                tf.summary.scalar('Eval/Features/mAP/Objects',
                                  self.features_map['Objects'], step=step)

    def save_and_log(self, epoch, step):

        message = "    - Epoch: " + \
            str(epoch) + "\n    - Step: " + str(step) + "\n    - Value: "

        if(self.meter_loss_max.val == self.meter_loss_max.min):
            value = message + str(self.meter_loss_max.max)
            self.logger.log_metric('loss', 'max', value)

            if('max' in self.save):
                model_name = os.path.join(self.path_save, "max_best_loss.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_acc_max.val == self.meter_acc_max.max):
            value = message + str(self.meter_acc_max.max)
            self.logger.log_metric('accuracy', 'max', value)

            if('max' in self.save):
                model_name = os.path.join(
                    self.path_save, "max_best_accuracy.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_map_max.val == self.meter_map_max.max):
            value = message + str(self.meter_map_max.max)
            self.logger.log_metric('precision', 'max', value)

            if('max' in self.save):
                model_name = os.path.join(
                    self.path_save, "max_best_precision.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_loss_mean.val == self.meter_loss_mean.min):
            value = message + str(self.meter_loss_mean.max)
            self.logger.log_metric('loss', 'mean', value)

            if('mean' in self.save):
                model_name = os.path.join(self.path_save, "mean_best_loss.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_acc_mean.val == self.meter_acc_mean.max):
            self.epoch_best_acc = epoch
            value = message + str(self.meter_acc_mean.max)
            self.logger.log_metric('accuracy', 'mean', value)

            if('mean' in self.save):
                model_name = os.path.join(
                    self.path_save, "mean_best_accuracy.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_map_mean.val == self.meter_map_mean.max):
            self.epoch_best_map = epoch
            value = message + str(self.meter_map_mean.max)
            self.logger.log_metric('precision', 'mean', value)

            if('mean' in self.save):
                model_name = os.path.join(
                    self.path_save, "mean_best_precision.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_loss_vote.val == self.meter_loss_vote.min):
            self.epoch_best_loss = epoch
            value = message + str(self.meter_loss_vote.max)
            self.logger.log_metric('loss', 'vote', value)

            if('vote' in self.save):
                model_name = os.path.join(self.path_save, "vote_best_loss.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_acc_vote.val == self.meter_acc_vote.max):
            self.epoch_best_acc = epoch
            value = message + str(self.meter_acc_vote.max)
            self.logger.log_metric('accuracy', 'vote', value)

            if('vote' in self.save):
                model_name = os.path.join(
                    self.path_save, "vote_best_accuracy.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if(self.meter_map_vote.val == self.meter_map_vote.max):
            self.epoch_best_map = epoch
            value = message + str(self.meter_map_vote.max)
            self.logger.log_metric('precision', 'vote', value)

            if('vote' in self.save):
                model_name = os.path.join(
                    self.path_save, "vote_best_precision.h5")
                self.model.save(model_name, overwrite=True,
                                include_optimizer=True, save_format='h5')

        if('none' not in self.save):
            model_name = os.path.join(self.path_save, "last_iteration.h5")
            self.model.save(model_name, overwrite=True,
                            include_optimizer=True, save_format='h5')

        self.time_total = time.time() - self.time_start
        str_time = get_time(self.time_total)

        self.logger.log_epochs(str(epoch))
        self.logger.log_time(str_time)
        self.logger.save()

    def build_AGN(self, fraction):

        # Get Model
        if(self.architecture == 'baseline'):
            self.model = get_model_baseline(
                features_list=self.features,
                size_hidden_state=self.size_hidden_state,
                feature2size=feature2size,
                num_classes=self.num_classes,
                num_steps=self.num_steps,
                dropout=self.rate_dropout
            )

        elif(self.architecture == 'edge_mlp'):
            self.model = get_model_edge_mlp(
                features_list=self.features,
                size_hidden_state=self.size_hidden_state,
                feature2size=feature2size,
                num_classes=self.num_classes,
                num_steps=self.num_steps,
                dropout=self.rate_dropout
            )

        else:
            print(">> [ERROR] Model architecture doesnt exist")
            sys.exit()

        # Optimizer
        if(self.optimizer_type == 'adam'):
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.rate_learning, amsgrad=True, name='adam')

        elif(self.optimizer_type == 'adamw'):
            self.optimizer = tfa.optimizers.AdamW(
                learning_rate=self.rate_learning, weight_decay=self.rate_weight, amsgrad=True, name='adamw')

            # Weight Decay Schedule
            self.wd_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.rate_weight,
                decay_steps=fraction,
                decay_rate=self.rate_decay,
                staircase=False
            )

        else:
            print(">> [ERROR] Optimizer type doesnt exists")
            sys.exit()

        # Learning Rate Schedule
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.rate_learning,
            decay_steps=fraction,
            decay_rate=self.rate_decay,
            staircase=False
        )

        # Reduction
        if(self.loss_reduction == 'mean'):
            self.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

        elif(self.loss_reduction == 'sum'):
            self.reduction = tf.keras.losses.Reduction.SUM

        elif(self.loss_reduction == 'none'):
            self.reduction = tf.keras.losses.Reduction.NONE

        else:
            print(">> [ERROR] Loss reduction type doesnt exists")
            sys.exit()

        # Loss
        if(self.loss_type == 'cross_entropy'):
            self.loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, reduction=self.reduction, name=self.loss_type)

        elif(self.loss_type == 'focal'):
            self.loss = tfa.losses.SigmoidFocalCrossEntropy(
                from_logits=True, reduction=self.reduction, name=self.loss_type)

        else:
            print(">> [ERROR] Loss type doesnt exists")
            sys.exit()

        # Balance
        if(self.class_balance == 'weights_1'):
            self.class_weights = class2weight[self.num_classes]

        elif(self.class_balance == 'weights_2'):
            self.class_weights = class2weightnew[self.num_classes]

        elif(self.class_balance == 'none'):
            self.class_weights = None

        else:
            print(">> [ERROR] Class balance type doesnt exists")
            sys.exit()

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy'],
            run_eagerly=self.eager
        )

    def train(self):

        print(">> [TRAIN]")

        print(">>   Setting random generator seed... ")

        if(self.seed.isnumeric()):
            tf.random.set_seed(int(self.seed))

        elif(self.seed != "random"):
            print(">> [ERROR] Incorrect random generator seed")
            sys.exit()

        print(">>   ... done!")
        print(">>   Preparing datasets...")

        num_train_samples, dataset_train = get_dataset(
            self.path_features, "train")
        num_test_samples, dataset_test = get_dataset(
            self.path_features, "test")
        dataset_test = parse_dataset(dataset_test)

        print(">>   ... done!")
        print(">>   Initializing network and logger...")

        fraction = num_train_samples * 0.1
        self.build_AGN(fraction)

        self.initiate_logger('train')
        self.logger.log_configs(self.configs)
        self.initiate_writer()

        print(">>   ... done!")
        print(">>   Starting the training...")

        epoch = 1
        self.time_start = time.time()

        while epoch <= self.num_epochs:

            dataset = shuffle_dataset(dataset_train, num_train_samples)
            dataset = parse_dataset(dataset)

            step = 1
            multiplier = (epoch - 1) * num_train_samples
            time_epoch = time.time()
            self.reset_metrics()

            for sample in dataset:

                inputs, labels = get_inputs(
                    sample, self.features, self.num_classes, normalize=self.norm)
                out_loss, out_accuracy = self.model.train_on_batch(
                    inputs,
                    labels,
                    sample_weight=None,
                    class_weight=self.class_weights,
                    reset_metrics=True
                )

                self.update_metrics(out_accuracy, out_loss)
                self.update_summary_train(step + multiplier)

                if(step % 100 == 0):
                    print(">>   [Epoch %d/%d|Step %d] LR: %f <> " % (epoch, self.num_epochs, step,
                          tf.keras.backend.eval(self.model.optimizer.learning_rate)), end="", flush=True)
                    if(self.optimizer_type == 'adamw'):
                        print("WD: %f <> " % (tf.keras.backend.eval(
                            self.model.optimizer.weight_decay)), end="", flush=True)
                    print("Loss: %f <> Accuracy: %f" %
                          (self.meter_loss.avg, self.meter_accuracy.avg))

                if(int(step % fraction) == 0):
                    self.eval(num_test_samples, dataset_test)
                    self.update_summary_eval(step + multiplier)
                    self.update_summary_features(step + multiplier)
                    self.save_and_log(epoch, step)

                tf.keras.backend.set_value(
                    self.model.optimizer.learning_rate, self.lr_schedule(step + multiplier))
                if(self.optimizer_type == 'adamw'):
                    tf.keras.backend.set_value(
                        self.model.optimizer.weight_decay, self.wd_schedule(step + multiplier))

                step += 1

            assert num_train_samples == (step - 1), \
                ">> [ERROR] Incorrect number of samples on training"

            self.meter_time.update(time.time() - time_epoch)

            print(">>   [Time]")
            print(">>     Total: " + get_time(time.time() - self.time_start))
            print(">>     Epoch: " + get_time(self.meter_time.val) +
                  " <> Mean: " + get_time(self.meter_time.avg))

            if(epoch == self.num_epochs):
                if((self.epoch_best_acc == epoch) or (self.epoch_best_map == epoch) or (self.epoch_best_loss == epoch)):
                    self.num_epochs += 1

            epoch += 1

        print(">>   ... training finished!")

    def eval(self, num_samples, dataset):

        print(">> [EVAL]")
        print(">>   Starting validation...")

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

        self.process_probs(dict_probs, all_labels, 'Validation')

        print(">>   ... validation finished!")
        print(">>   Back to training...")

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

            if(key == 'Max'):
                self.meter_loss_max.update(scores['log_loss'])
                self.meter_acc_max.update(scores['accuracy'])
                self.meter_map_max.update(scores['mean_average_precision'])

            elif(key == 'Mean'):
                self.meter_loss_mean.update(scores['log_loss'])
                self.meter_acc_mean.update(scores['accuracy'])
                self.meter_map_mean.update(scores['mean_average_precision'])

            elif(key == 'Vote'):
                self.meter_loss_vote.update(scores['log_loss'])
                self.meter_acc_vote.update(scores['accuracy'])
                self.meter_map_vote.update(scores['mean_average_precision'])

            else:
                self.features_loss[key] = scores['log_loss']
                self.features_acc[key] = scores['accuracy']
                self.features_map[key] = scores['mean_average_precision']
