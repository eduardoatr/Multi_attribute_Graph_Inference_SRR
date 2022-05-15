import numpy as np
import psutil as ps


def get_time(time):
    hours, rem = divmod(time, 3600)
    minutes, seconds = divmod(rem, 60)
    str_time = "{:0>2}:{:0>2}:{:0>2}".format(
        int(hours), int(minutes), int(seconds))

    return str_time


def get_mem(self):
    used_mem = ps.virtual_memory().used
    str_mem = "Used Memory: {}MB".format(used_mem/1024/1024)

    return str_mem


def get_probs(probs, max, mean, vote):

    # Get probs shape
    size_features, size_class = probs.shape

    # Get mean probs
    probs_mean = np.mean(probs, axis=0)

    # Get max probs
    lin, _ = np.unravel_index(np.argmax(probs, axis=None), probs.shape)
    probs_max = probs[lin, :]

    # Get vote probs
    votes = np.argmax(probs, axis=1)
    probs_vote = np.zeros(size_class, dtype=np.float32)

    for winner in votes:
        probs_vote[winner] += (1/np.float32(size_features))

    max.append(probs_max)
    mean.append(probs_mean)
    vote.append(probs_vote)


def get_feature_probs(features, probs, features_probs, sample):

    # Get probs shape
    size_features, size_class = probs.shape

    # To keep track of each output
    index = 0

    # Metrics for feature type using MEAN
    for feature in features:

        temp = []
        data = sample[feature].numpy()
        l, c = data.shape

        for i in range(l):
            temp.append(probs[index])
            index += 1

        if(len(temp) == 0):
            temp.append(np.zeros(size_class, dtype=np.float32))

        mean = np.mean(np.asarray(temp), axis=0)

        features_probs[feature].append(mean)

    assert index == size_features, \
        ">> [ERROR] Problem on getting features probabilities"
