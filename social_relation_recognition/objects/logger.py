import os


class Logger(object):

    def __init__(self, name, save_path, type, num_labels=-1):

        self.type_list = ['patches', 'features', 'predicts', 'train']

        assert isinstance(name, str), \
            ">>   [LOGGER] Name is not a string"

        assert os.path.isdir(save_path), \
            ">>   [LOGGER] Save path is not a directory"

        assert type in self.type_list, \
            ">>   [LOGGER] Wrong logger type"

        self.name = name
        self.save_path = save_path
        self.type = type

        print(">>   [LOGGER] Creating log on: " + self.save_path)

        if((self.type != 'predicts') and (self.type != 'train')):

            assert num_labels > 0, \
                ">>   [LOGGER] Number of labels must be positive"

            self.num_labels = num_labels

            self.num_dir = 0
            self.num_files = 0

            self.count_label = {}
            for i in range(self.num_labels):
                self.count_label[i] = 0

            self.count_split = {
                'test': 0,
                'train': 0,
                'eval': 0
            }

        if(self.type == 'patches'):
            self.count_patch = {
                'body': 0,
                'face': 0,
                'relation': 0,
                'context': 0,
                'object': 0
            }

        if(self.type == 'features'):
            self.count_feature = {
                'objects_attention': 0,
                'context_emotion': 0,
                'context_activity': 0,
                'body_activity': 0,
                'body_gender': 0,
                'body_age': 0,
                'body_clothing': 0
            }

        if(self.type == 'predicts'):
            self.predicts = {}

        if(self.type == 'train'):

            self.metrics = ['loss', 'accuracy', 'precision']
            self.modes = ['max', 'mean', 'vote']

            self.configs = []
            self.total_epochs = "Not logged yet"
            self.total_time = "Not logged yet"
            self.best_loss_max = "    - Not logged yet"
            self.best_loss_mean = "    - Not logged yet"
            self.best_loss_vote = "    - Not logged yet"
            self.best_accuracy_max = "    - Not logged yet"
            self.best_accuracy_mean = "    - Not logged yet"
            self.best_accuracy_vote = "    - Not logged yet"
            self.best_precision_max = "    - Not logged yet"
            self.best_precision_mean = "    - Not logged yet"
            self.best_precision_vote = "    - Not logged yet"

    def log_dir(self):

        assert (self.type != 'predicts') and (self.type != 'train'), \
            ">>   [LOGGER] This logger doesn't have this function"

        self.num_dir += 1

    def log_patch(self, split, label, patch):

        assert self.type == 'patches', \
            ">>   [LOGGER] This logger is not a patcher"

        assert split in self.count_split, \
            ">>   [LOGGER] Split type doesn't exists"

        assert label in self.count_label, \
            ">>   [LOGGER] Label number doesn't exists"

        assert patch in self.count_patch, \
            ">>   [LOGGER] Patch type doesn't exists"

        self.num_files += 1
        self.count_split[split] += 1
        self.count_label[label] += 1
        self.count_patch[patch] += 1

    def log_feature(self, split, label, feature):

        assert self.type == 'features', \
            ">>   [LOGGER] This logger is not a featurer"

        assert split in self.count_split, \
            ">>   [LOGGER] Split type doesn't exists"

        assert label in self.count_label, \
            ">>   [LOGGER] Label number doesn't exists"

        assert feature in self.count_feature, \
            ">>   [LOGGER] Feature type doesn't exists"

        self.num_files += 1
        self.count_split[split] += 1
        self.count_label[label] += 1
        self.count_feature[feature] += 1

    def log_predict(self, relation, predict):

        assert self.type == 'predicts', \
            ">>   [LOGGER] This logger is not a predictor"

        self.predicts[relation] = predict

    def log_configs(self, configurations):

        assert self.type == 'train', \
            ">>   [LOGGER] This logger is not a trainer"

        self.configs = configurations

    def log_metric(self, metric, mode, value):

        assert self.type == 'train', \
            ">>   [LOGGER] This logger is not a trainer"

        assert metric in self.metrics, \
            ">>   [LOGGER] Invalid metric"

        assert mode in self.modes, \
            ">>   [LOGGER] Invalid mode"

        if(metric == 'loss'):
            if(mode == 'max'):
                self.best_loss_max = value

            elif(mode == 'mean'):
                self.best_loss_mean = value

            else:
                self.best_loss_vote = value

        elif(metric == 'accuracy'):
            if(mode == 'max'):
                self.best_accuracy_max = value

            elif(mode == 'mean'):
                self.best_accuracy_mean = value

            else:
                self.best_accuracy_vote = value

        else:
            if(mode == 'max'):
                self.best_precision_max = value

            elif(mode == 'mean'):
                self.best_precision_mean = value

            else:
                self.best_precision_vote = value

    def log_epochs(self, epochs):

        assert self.type == 'train', \
            ">>   [LOGGER] This logger is not a trainer"

        self.total_epochs = epochs

    def log_time(self, time):

        assert self.type == 'train', \
            ">>   [LOGGER] This logger is not a trainer"

        self.total_time = time

    def save(self):

        print(">>   [LOGGER] Logging... ")

        if(self.type == 'train'):

            path = os.path.join(
                self.save_path, "configs_" + self.name + ".yaml")

            with open(path, "w") as file:
                file.write("#[CONFIGURATIONS]\n")
                for conf in self.configs:
                    file.write(conf + "\n")

            path = os.path.join(
                self.save_path, "scores_" + self.name + ".yaml")

            with open(path, "w") as file:
                file.write("#[TRAINING]\n")
                file.write("Epochs: " + self.total_epochs + "\n")
                file.write("Time: " + self.total_time + "\n")

                file.write("#[BEST SCORES]\n")
                file.write("Loss:\n")
                file.write("  - Max:\n")
                file.write(self.best_loss_max + "\n")

                file.write("  - Mean:\n")
                file.write(self.best_loss_mean + "\n")

                file.write("  - Vote:\n")
                file.write(self.best_loss_vote + "\n")

                file.write("Accuracy:\n")
                file.write("  - Max:\n")
                file.write(self.best_accuracy_max + "\n")

                file.write("  - Mean:\n")
                file.write(self.best_accuracy_mean + "\n")

                file.write("  - Vote:\n")
                file.write(self.best_accuracy_vote + "\n")

                file.write("Precision:\n")
                file.write("  - Max:\n")
                file.write(self.best_precision_max + "\n")

                file.write("  - Mean:\n")
                file.write(self.best_precision_mean + "\n")

                file.write("  - Vote:\n")
                file.write(self.best_precision_vote + "\n")

        else:

            path = os.path.join(self.save_path, "logs_" + self.name + ".yaml")

            with open(path, "w") as file:
                if(self.type != 'predicts'):
                    file.write("#[TOTAL]\n")
                    file.write("dir: %d\n" % self.num_dir)
                    file.write("files: %d\n" % self.num_files)

                    file.write("#[SPLITS]\n")
                    for key in self.count_split:
                        file.write("%s: %d\n" % (key, self.count_split[key]))

                    file.write("#[LABELS]\n")
                    for i in range(self.num_labels):
                        file.write("%d: %d\n" % (i, self.count_label[i]))

                if(self.type == 'patches'):
                    file.write("#[PATCHES]\n")
                    for key in self.count_patch:
                        file.write(str(key) + ": %d\n" % self.count_patch[key])

                if(self.type == 'features'):
                    file.write("#[FEATURES]\n")
                    for key in self.count_feature:
                        file.write(str(key) + ": %d\n" %
                                   self.count_feature[key])

                if(self.type == 'predicts'):
                    file.write("#[PREDITCS]\n")
                    for key in self.predicts:
                        file.write(str(key) + ": %d\n" % self.predicts[key])

        print(">>   [LOGGER] ... done!")
