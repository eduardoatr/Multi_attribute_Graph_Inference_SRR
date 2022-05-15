import argparse
import os
import sys
import time

import cv2
import numpy as np
import objects.logger as l

# Dictionaries & Lists
partitions_list = [
    'train',
    'eval',
    'test'
]

type2class = {
    'coarse': 3,
    'domain': 5,
    'fine': 6,
    'relation': 16
}

feature2model = {
    "objects_attention": "SENet.caffemodel",
    "context_emotion": "group_scene.caffemodel",
    "context_activity": "SituationCrf.caffemodel.h5",
                        "body_activity": "SituationCrf_body.caffemodel.h5",
                        "body_age": "body_age.caffemodel",
                        "body_clothing": "body_clothing.caffemodel",
                        "body_gender": "body_gender.caffemodel"
}

model2protocol = {
    "SENet.caffemodel": "SENet.prototxt",
                        "group_scene.caffemodel": "group_scene.prototxt",
                        "SituationCrf.caffemodel.h5": "SituationCrf.prototxt",
                        "SituationCrf_body.caffemodel.h5": "SituationCrf.prototxt",
                        "body_age.caffemodel": "double_stream.prototxt",
                        "body_clothing.caffemodel": "double_stream.prototxt",
                        "body_gender.caffemodel": "double_stream.prototxt",
}

model2layer = {
    "SENet.caffemodel": "pool5/7x7_s1",
    "group_scene.caffemodel": "global_pool",
    "SituationCrf.caffemodel.h5": "fc7",
    "SituationCrf_body.caffemodel.h5": "fc7",
    "body_age.caffemodel": "fc7",
    "body_clothing.caffemodel": "fc7",
    "body_gender.caffemodel": "fc7"
}

model2feature = {
    "SENet.caffemodel": "objects",
                        "group_scene.caffemodel": "context",
                        "SituationCrf.caffemodel.h5": "context",
                        "SituationCrf_body.caffemodel.h5": "bodies",
                        "body_age.caffemodel": "bodies",
                        "body_clothing.caffemodel": "bodies",
                        "body_gender.caffemodel": "bodies"
}

model2attribute = {
    "SENet.caffemodel": "attention",
                        "group_scene.caffemodel": "emotion",
                        "SituationCrf.caffemodel.h5": "activity",
                        "SituationCrf_body.caffemodel.h5": "activity",
                        "body_age.caffemodel": "age",
                        "body_clothing.caffemodel": "clothing",
                        "body_gender.caffemodel": "gender"
}


def caffe_import(model):

    if(model == 'SENet.caffemodel'):
        sys.path.insert(0, '/home/eduardo/Code/NU/Caffe/Caffe_SENet/python')

    elif(model == 'SituationCrf.caffemodel.h5'):
        sys.path.insert(
            0, '/home/eduardo/Code/NU/Caffe/Caffe_SituationCrf/python')

    else:
        sys.path.insert(0, '/home/eduardo/Code/NU/Caffe/Caffe_default/python')

    import caffe

    return caffe


def initialize_transformer(caffe, protocol):

    if(protocol == "double_stream.prototxt"):
        shape = (1, 3, 227, 227)
        transformer = caffe.io.Transformer({'data': shape})
        channel_mean = np.zeros((3, 227, 227))

    else:
        shape = (1, 3, 224, 224)
        transformer = caffe.io.Transformer({'data': shape})
        channel_mean = np.zeros((3, 224, 224))

    if(protocol == "group_scene.prototxt"):
        image_mean = [90, 100, 128]

    else:
        image_mean = [104, 117, 123]  # ImageNet mean

    for channel_index, mean_val in enumerate(image_mean):
        channel_mean[channel_index, ...] = mean_val

    transformer.set_mean('data', channel_mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_transpose('data', (2, 0, 1))

    return transformer


def build_structure(args):

    logger = l.Logger(str(args.cue), args.save_dir,
                      type2class[args.type], 'features')

    root_path = args.save_dir

    if not os.path.exists(root_path):
        os.mkdir(root_path)
        print(">> Directory ( " + root_path + " ) created")
        logger.log_dir()

    for partition in partitions_list:

        partition_path = os.path.join(root_path, partition)

        if not os.path.exists(partition_path):
            os.mkdir(partition_path)
            print(">> Directory ( " + partition_path + " ) created")
            logger.log_dir()

        for i in range(type2class[args.type]):

            class_path = os.path.join(partition_path, str(i))

            if not os.path.exists(class_path):
                os.mkdir(class_path)
                print(">> Directory ( " + class_path + " ) created")
                logger.log_dir()

    logger.save()


def extract_features(args):

    logger_features = l.Logger(
        str(args.cue), args.save_dir, type2class[args.type], 'features')
    logger_predicts = None

    model = feature2model[args.cue]
    protocol = model2protocol[model]
    layer = model2layer[model]
    feature = model2feature[model]
    attribute = model2attribute[model]

    print(">> [Importing] Model: " + model +
          " <<>> Protocol: " + protocol + " ...")

    caffe = caffe_import(model)
    caffe.set_mode_gpu()

    transformer_RGB = initialize_transformer(caffe, protocol)

    protocol_path = os.path.join(args.model_dir, 'Caffe_protocol', protocol)
    model_path = os.path.join(args.model_dir, 'Caffe_model', model)

    net = caffe.Net(protocol_path, caffe.TEST, weights=model_path)

    if("prob" in net.blobs):
        logger_predicts = l.Logger(
            str(args.cue) + "predicts", args.save_dir, type2class[args.type], 'predicts')

    for partition in partitions_list:

        print(">> [" + partition + "] extracting " +
              feature + " " + attribute + " features...")

        for i in range(type2class[args.type]):

            class_path = os.path.join(args.data_dir, partition, str(i))

            for image_name in sorted(os.listdir(class_path)):

                save_folder = os.path.join(
                    args.save_dir, partition, str(i), image_name)

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                save_folder = os.path.join(
                    save_folder, feature + "_" + attribute)

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                if(protocol != "double_stream.prototxt"):

                    for patch_image in sorted(os.listdir(os.path.join(class_path, image_name, feature))):

                        feature_name = patch_image.split(
                            '.')[-2] + "_" + attribute + '.npy'

                        save_feature_path = os.path.join(
                            save_folder, feature_name)

                        if os.path.isfile(save_feature_path):
                            continue

                        patch_image = os.path.join(
                            class_path, image_name, feature, patch_image)

                        input_im = caffe.io.load_image(patch_image)

                        if input_im.shape[0] != 256 or input_im.shape[1] != 256:
                            input_im = caffe.io.resize_image(
                                input_im, (256, 256))

                        net.blobs['data'].data[...] = transformer_RGB.preprocess(
                            'data', input_im)

                        probs = net.forward()

                        #############################################################################################
                        ####                             DEBUG MODE: begin                                       ####
                        #############################################################################################

                        #print(">> [Blobs] {}\n>> [Params] {}".format(net.blobs.keys(), net.params.keys()))

                        # Blobs and Param for important layers
                        #print(">> [data] Shape: {}".format(net.blobs['data'].data[0].shape))
                        #print(">> [data] " + "Mean: %f" % net.blobs['data'].data[0].mean())
                        #print(">> [" + layer + "] Shape: {}".format(net.blobs[layer].data[0].shape))
                        #print(">> [" + layer + "] " + "Mean: %f" % net.blobs[layer].data[0].mean())

                        # if not(logger_predicts is None):
                        #    print(">> [prob] Shape: {}".format(net.blobs['prob'].data[0].shape))
                        #    print(">> [prob] " + "Mean: %f" % net.blobs['prob'].data[0].mean())
                        #    print(">> [prob] Label %d " % np.argmax(probs))

                        #im = transformer_RGB.deprocess('data', net.blobs['data'].data[0])
                        #imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        #cv2.imshow("Imagem de Entrada", imRGB)
                        # cv2.waitKey(0)

                        #im2 = net.blobs['data'].data[0]
                        #im2 = im2.transpose()
                        #im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                        #cv2.imshow("Imagem de Entrada Processada", im2RGB)
                        # cv2.waitKey(0)

                        # time.sleep(8)

                        #############################################################################################
                        ####                               DEBUG MODE: end                                       ####
                        #############################################################################################

                        if not(logger_predicts is None):
                            predict = np.argmax(probs)
                            logger_predicts.log_predict(patch_image, predict)

                        extracted_features = net.blobs[layer].data[0]

                        np.save(save_feature_path, extracted_features)
                        logger_features.log_feature(partition, i, args.cue)

                else:

                    dir_path = os.path.join(class_path, image_name, 'bodies')

                    feature1_name = attribute + '_1.npy'
                    feature2_name = attribute + '_2.npy'

                    save_feature1_path = os.path.join(
                        save_folder, feature1_name)
                    save_feature2_path = os.path.join(
                        save_folder, feature2_name)

                    if ((os.path.isfile(save_feature1_path)) and (os.path.isfile(save_feature2_path))):
                        continue

                    body1_image = os.path.join(dir_path, "body_1.jpg")
                    body2_image = os.path.join(dir_path, "body_2.jpg")

                    input1_im = caffe.io.load_image(body1_image)
                    input2_im = caffe.io.load_image(body2_image)

                    if input1_im.shape[0] != 256 or input1_im.shape[1] != 256:
                        input1_im = caffe.io.resize_image(
                            input1_im, (256, 256))

                    if input2_im.shape[0] != 256 or input2_im.shape[1] != 256:
                        input2_im = caffe.io.resize_image(
                            input2_im, (256, 256))

                    net.blobs['data'].data[...] = transformer_RGB.preprocess(
                        'data', input1_im)
                    net.blobs['data_1'].data[...] = transformer_RGB.preprocess(
                        'data', input2_im)

                    probs = net.forward()

                    #############################################################################################
                    ####                             DEBUG MODE: begin                                       ####
                    #############################################################################################

                    #print(">> [Blobs] {}\n>> [Params] {}".format(net.blobs.keys(), net.params.keys()))

                    # Blobs and Param for important layers
                    #print(">> [data] " + "Shape: {}".format(net.blobs['data'].data[0].shape))
                    #print(">> [data] " + "Mean: %f" % net.blobs['data'].data[0].mean())
                    #print(">> [data] " + "Shape: {}".format(net.blobs['data_1'].data[0].shape))
                    #print(">> [data_1] " + "Mean: %f" % net.blobs['data_1'].data[0].mean())
                    #print(">> [data |" + layer + "] Shape: {}".format(net.blobs[layer].data[0].shape))
                    #print(">> [data |" + layer + "] " + "Mean: %f" % net.blobs[layer].data[0].mean())
                    #print(">> [data_1 |" + layer + "] Shape: {}".format(net.blobs[layer].data[1].shape))
                    #print(">> [data_1 |" + layer + "] " + "Mean: %f" % net.blobs[layer].data[1].mean())

                    # if not(logger_predicts is None):
                    #    print(">> [prob] Shape: {}".format(net.blobs['prob'].data[0].shape))
                    #    print(">> [prob] " + "Mean: %f" % net.blobs['prob'].data[0].mean())
                    #    print(">> [prob] Label %d " % np.argmax(probs))

                    #im = transformer_RGB.deprocess('data', net.blobs['data'].data[0])
                    #imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    #cv2.imshow("Imagem de Entrada 1", imRGB)
                    # cv2.waitKey(0)

                    #im = transformer_RGB.deprocess('data', net.blobs['data_1'].data[0])
                    #imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    #cv2.imshow("Imagem de Entrada 2", imRGB)
                    # cv2.waitKey(0)

                    #im2 = net.blobs['data'].data[0]
                    #im2 = im2.transpose()
                    #im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                    #cv2.imshow("Imagem de Entrada 1 Processada", im2RGB)
                    # cv2.waitKey(0)

                    #im2 = net.blobs['data_1'].data[0]
                    #im2 = im2.transpose()
                    #im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                    #cv2.imshow("Imagem de Entrada 2 Processada", im2RGB)
                    # cv2.waitKey(0)

                    # time.sleep(8)

                    #############################################################################################
                    ####                               DEBUG MODE: end                                       ####
                    #############################################################################################

                    if not(logger_predicts is None):
                        predict = np.argmax(probs)
                        logger_predicts.log_predict(
                            os.path.join(class_path, image_name), predict)

                    extracted_features1 = net.blobs[layer].data[0]
                    extracted_features2 = net.blobs[layer].data[1]

                    np.save(save_feature1_path, extracted_features1)
                    np.save(save_feature2_path, extracted_features2)

                    logger_features.log_feature(partition, i, args.cue)
                    logger_features.log_feature(partition, i, args.cue)

        print(">> ... done!")

        logger_features.save()
        if not(logger_predicts is None):
            logger_predicts.save()

    del net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extrace features')
    parser.add_argument('--data_dir', type=str, metavar='DIR',
                        help='path to the input data')
    parser.add_argument('--save_dir', type=str, metavar='DIR',
                        help='path to the extract features')
    parser.add_argument('--model_dir', type=str, metavar='DIR',
                        help='path to caffe models')
    parser.add_argument('--type',  type=str, help='type of dataset', choices=[
        'fine',
        'coarse',
        'relation',
        'domain'
    ])
    parser.add_argument('--cue', type=str, help='type of feature', choices=[
        'objects_attention',
        'context_emotion',
        'context_activity',
        'body_activity',
        'body_gender',
        'body_age',
        'body_clothing'
    ])

    args = parser.parse_args()
    assert args.data_dir != args.save_dir

    build_structure(args)
    extract_features(args)
