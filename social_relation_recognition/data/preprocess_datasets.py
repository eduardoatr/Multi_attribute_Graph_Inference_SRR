import argparse
import json
import os

import objects.logger as l
from PIL import Image

# Dictionaries & Lists
type2class_PIPA = {
    'relation': 16
}

type2class_PISC = {
    'coarse': 3,
    'fine': 6
}

root_list = [
    'relations',
    'patches'
]

partitions_list = [
    'train',
    'eval',
    'test'
]


def dir_path(string):
    assert os.path.isdir(string), \
        ">> "+string+" is not a directory!"

    return string


def build_structure_PIPA(args):

    for type in type2class_PIPA:

        logger = l.Logger("Structure_" + str(type),
                          args.save_dir, type2class_PIPA[type], 'patches')

        type_path = os.path.join(args.save_dir, type)

        if not os.path.exists(type_path):
            os.mkdir(type_path)
            print(">> Directory ( " + type_path + " ) created")
            logger.log_dir()

        for root in root_list:

            root_path = os.path.join(type_path, root)

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

                for i in range(type2class_PIPA[type]):

                    class_path = os.path.join(partition_path, str(i))

                    if not os.path.exists(class_path):
                        os.mkdir(class_path)
                        print(">> Directory ( " + class_path + " ) created")
                        logger.log_dir()

        logger.save()


def build_structure_PISC(args):

    for type in type2class_PISC:

        logger = l.Logger("Structure_" + str(type),
                          args.save_dir, type2class_PISC[type], 'patches')

        type_path = os.path.join(args.save_dir, type)

        if not os.path.exists(type_path):
            os.mkdir(type_path)
            print(">> Directory ( " + type_path + " ) created")
            logger.log_dir()

        for root in root_list:

            root_path = os.path.join(type_path, root)

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

                for i in range(type2class_PISC[type]):

                    class_path = os.path.join(partition_path, str(i))

                    if not os.path.exists(class_path):
                        os.mkdir(class_path)
                        print(">> Directory ( " + class_path + " ) created")
                        logger.log_dir()

        logger.save()


def build_PIPA_relations_patches(args):

    for type in type2class_PIPA:

        logger = l.Logger("Relations_" + str(type),
                          args.save_dir, type2class_PIPA[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'relations')
        patches_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition +
                  "] Extracting relation and generating patches folders")
            print(">> ...")

            list_path = os.path.join(
                args.data_dir, 'annotations', 'single_body1_'+partition+'_'+str(type2class_PIPA[type])+'.txt',)

            list = open(list_path)

            lines = list.readlines()

            for relation in range(len(lines)):

                line = lines[relation]

                line = line.strip().split()

                img_path = line[0]
                label = line[-1]

                line = img_path.strip().split("_")

                img_part1 = line[-2]
                img_part2 = line[-1]

                img_path = img_part1 + "_" + img_part2

                img_path = os.path.join(
                    args.data_dir, 'images', 'all_full_image', img_path)

                img = Image.open(img_path).convert('RGB')

                assert (img is not None) and (img.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                img_name = "relation_" + str(relation) + ".jpg"

                dir_name = os.path.join(root_path, partition, label, img_name)

                img.save(dir_name)
                logger.log_patch(partition, int(label), 'relation')

                dir_name_patches = os.path.join(
                    patches_path, partition, label, "relation_" + str(relation))

                if not os.path.exists(dir_name_patches):
                    os.mkdir(dir_name_patches)
                    logger.log_dir()

            print(">> ...done!")

            list.close()

        logger.save()


def build_PISC_relations_patches(args):

    for type in type2class_PISC:

        logger = l.Logger("Relations_" + str(type),
                          args.save_dir, type2class_PISC[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'relations')
        patches_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition +
                  "] Extracting relation and generating patches folders")
            print(">> ...")

            list_path = os.path.join(
                args.data_dir, 'lists', 'PISC_' + type + '_level_' + partition + '.txt',)

            list = open(list_path)

            lines = list.readlines()

            for relation in range(len(lines)):

                line = lines[relation]

                line = line.strip().split()

                img_name = line[0]
                label = line[-1]

                img_path = os.path.join(
                    args.data_dir, 'images', 'all_full_image', img_name)

                img = Image.open(img_path).convert('RGB')

                assert (img is not None) and (img.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                img_name = "relation_" + str(relation) + ".jpg"

                dir_name = os.path.join(root_path, partition, label, img_name)

                img.save(dir_name)
                logger.log_patch(partition, int(label), 'relation')

                dir_name_patches = os.path.join(
                    patches_path, partition, label, "relation_" + str(relation))

                if not os.path.exists(dir_name_patches):
                    os.mkdir(dir_name_patches)
                    logger.log_dir()

            print(">> ...done!")

            list.close()

        logger.save()


def build_PIPA_bodies(args):

    for type in type2class_PIPA:

        logger = l.Logger("Bodies_" + str(type), args.save_dir,
                          type2class_PIPA[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition + "] Extracting body patches")
            print(">> ...")

            person1_path = os.path.join(
                args.data_dir, 'annotations', 'single_body1_'+partition+'_'+str(type2class_PIPA[type])+'.txt',)
            person2_path = os.path.join(
                args.data_dir, 'annotations', 'single_body2_'+partition+'_'+str(type2class_PIPA[type])+'.txt',)

            person1 = open(person1_path)
            person2 = open(person2_path)

            lines1 = person1.readlines()
            lines2 = person2.readlines()

            for relation in range(len(lines1)):

                line1 = lines1[relation]
                line2 = lines2[relation]

                line1 = line1.strip().split()
                line2 = line2.strip().split()

                img1_path = line1[0]
                img2_path = line2[0]

                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')

                assert (img1 is not None) and (img1.size > 0), \
                    ">> [ERROR] Image ( " + img1_path + " ) is empty"

                assert (img2 is not None) and (img2.size > 0), \
                    ">> [ERROR] Image ( " + img2_path + " ) is empty"

                label = line1[-1]

                dir_name = os.path.join(
                    root_path, partition, label, 'relation_'+str(relation), 'bodies')

                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                    logger.log_dir()

                img1.save(dir_name+"/body_1.jpg")
                logger.log_patch(partition, int(label), 'body')

                img2.save(dir_name+"/body_2.jpg")
                logger.log_patch(partition, int(label), 'body')

            print(">> ...done!")

            person1.close()
            person2.close()

        logger.save()


def build_PISC_bodies(args):

    for type in type2class_PISC:

        logger = l.Logger("Bodies_" + str(type), args.save_dir,
                          type2class_PISC[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition + "] Extracting body patches")
            print(">> ...")

            list_path = os.path.join(
                args.data_dir, 'lists', 'PISC_' + type + '_level_' + partition + '.txt',)

            list = open(list_path)

            lines = list.readlines()

            for relation in range(len(lines)):

                line = lines[relation]

                line = line.strip().split()

                img_name = line[0]
                label = line[-1]

                img_path = os.path.join(
                    args.data_dir, 'images', 'all_full_image', img_name)

                img = Image.open(img_path).convert('RGB')

                assert (img is not None) and (img.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                p1x1 = int(line[1])
                p1y1 = int(line[2])
                p1x2 = int(line[3])
                p1y2 = int(line[4])

                p2x1 = int(line[5])
                p2y1 = int(line[6])
                p2x2 = int(line[7])
                p2y2 = int(line[8])

                body1 = img.crop((p1x1, p1y1, p1x2, p1y2))
                body2 = img.crop((p2x1, p2y1, p2x2, p2y2))

                assert (body1 is not None) and (body1.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                assert (body2 is not None) and (body2.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                dir_name = os.path.join(
                    root_path, partition, label, 'relation_'+str(relation), 'bodies')

                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                    logger.log_dir()

                body1.save(dir_name+"/body_1.jpg")
                logger.log_patch(partition, int(label), 'body')

                body2.save(dir_name+"/body_2.jpg")
                logger.log_patch(partition, int(label), 'body')

            print(">> ...done!")

            list.close()

        logger.save()


def build_PIPA_objects(args):

    for type in type2class_PIPA:

        logger = l.Logger("Objects_" + str(type), args.save_dir,
                          type2class_PIPA[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition +
                  "] Extracting object patches")
            print(">> ...")

            list_path = os.path.join(
                args.data_dir, 'annotations', 'single_body1_'+partition+'_'+str(type2class_PIPA[type])+'.txt',)

            list = open(list_path)

            lines = list.readlines()

            for relation in range(len(lines)):

                line = lines[relation]

                line = line.strip().split()

                label = line[-1]

                dir_name = os.path.join(
                    root_path, partition, label, 'relation_'+str(relation), 'objects')

                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                    logger.log_dir()

                line = line[0]

                line = line.strip().split("_")

                img_part1 = line[-2]
                img_part2 = line[-1]

                img_name = img_part1 + "_" + img_part2

                img_path = os.path.join(
                    args.data_dir, 'images', 'all_full_image', img_name)

                img_part2 = img_part2.split(".")[0]

                boxes_path = img_part1 + "_" + img_part2 + '.json'

                boxes_path = os.path.join(args.data_dir, 'objects', boxes_path)

                boxes_data = json.load(open(boxes_path))

                object_boxes = boxes_data["bboxes"]

                img = Image.open(img_path).convert('RGB')

                assert (img is not None) and (img.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                for i, box in enumerate(object_boxes):

                    if(i == 0):
                        continue

                    xstart = int(box[0])
                    ystart = int(box[1])
                    xend = int(box[2])
                    yend = int(box[3])

                    object_crop = img.crop((xstart, ystart, xend, yend))

                    assert (object_crop is not None) and (object_crop.size > 0), \
                        ">> [ERROR] Relation " + \
                        str(relation) + " object " + str(i) + " is empty"

                    object_crop.save(dir_name + "/object_" + str(i-1) + ".jpg")
                    logger.log_patch(partition, int(label), 'object')

            print(">> ...done!")

            list.close()

        logger.save()


def build_PISC_objects(args):

    for type in type2class_PISC:

        logger = l.Logger("Objects_" + str(type), args.save_dir,
                          type2class_PISC[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition +
                  "] Extracting object patches")
            print(">> ...")

            list_path = os.path.join(
                args.data_dir, 'lists', 'PISC_' + type + '_level_' + partition + '.txt',)

            list = open(list_path)

            lines = list.readlines()

            for relation in range(len(lines)):

                line = lines[relation]

                line = line.strip().split()

                img_name = line[0]
                label = line[-1]

                dir_name = os.path.join(
                    root_path, partition, label, 'relation_'+str(relation), 'objects')

                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                    logger.log_dir()

                img_path = os.path.join(
                    args.data_dir, 'images', 'all_full_image', img_name)

                img_name = img_name.split(".")[0]

                boxes_path = img_name + '.json'

                boxes_path = os.path.join(args.data_dir, 'objects', boxes_path)

                boxes_data = json.load(open(boxes_path))

                object_boxes = boxes_data["bboxes"]

                img = Image.open(img_path).convert('RGB')

                assert (img is not None) and (img.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                for i, box in enumerate(object_boxes):

                    if(i == 0):
                        continue

                    xstart = int(box[0])
                    ystart = int(box[1])
                    xend = int(box[2])
                    yend = int(box[3])

                    object_crop = img.crop((xstart, ystart, xend, yend))

                    assert (object_crop is not None) and (object_crop.size > 0), \
                        ">> [ERROR] Relation " + \
                        str(relation) + " object " + str(i) + " is empty"

                    object_crop.save(dir_name + "/object_" + str(i-1) + ".jpg")
                    logger.log_patch(partition, int(label), 'object')

            print(">> ...done!")

            list.close()

        logger.save()


def build_PIPA_context(args):

    for type in type2class_PIPA:

        logger = l.Logger("Context_" + str(type), args.save_dir,
                          type2class_PIPA[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition +
                  "] Extracting context patches")
            print(">> ...")

            list_path = os.path.join(
                args.data_dir, 'lists', 'PIPA_relation_'+partition+'.txt',)

            list = open(list_path)

            lines = list.readlines()

            for relation in range(len(lines)):

                line = lines[relation]

                line = line.strip().split()

                label = line[-1]

                img_name = line[0]

                img_path = os.path.join(
                    args.data_dir, 'images', 'all_full_image', img_name)

                img = Image.open(img_path).convert('RGB')

                assert (img is not None) and (img.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                p1x1 = int(line[1])
                p1y1 = int(line[2])
                p1x2 = int(line[3])
                p1y2 = int(line[4])

                p2x1 = int(line[5])
                p2y1 = int(line[6])
                p2x2 = int(line[7])
                p2y2 = int(line[8])

                pux1 = min(p1x1, p2x1)
                puy1 = min(p1y1, p2y1)
                pux2 = max(p1x2, p2x2)
                puy2 = max(p1y2, p2y2)

                context_crop = img.crop((pux1, puy1, pux2, puy2))

                assert (context_crop is not None) and (context_crop.size > 0), \
                    ">> [ERROR] Relation " + \
                    str(relation) + " context is empty"

                dir_name = os.path.join(
                    root_path, partition, label, 'relation_'+str(relation), 'context')

                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                    logger.log_dir()

                context_crop.save(dir_name+"/context.jpg")
                logger.log_patch(partition, int(label), 'context')

            print(">> ...done!")

            list.close()

        logger.save()


def build_PISC_context(args):

    for type in type2class_PISC:

        logger = l.Logger("Context_" + str(type), args.save_dir,
                          type2class_PISC[type], 'patches')

        root_path = os.path.join(args.save_dir, type, 'patches')

        for partition in partitions_list:

            print(">> [" + type + "|" + partition +
                  "] Extracting context patches")
            print(">> ...")

            list_path = os.path.join(
                args.data_dir, 'lists', 'PISC_' + type + '_level_' + partition + '.txt',)

            list = open(list_path)

            lines = list.readlines()

            for relation in range(len(lines)):

                line = lines[relation]

                line = line.strip().split()

                img_name = line[0]
                label = line[-1]

                img_path = os.path.join(
                    args.data_dir, 'images', 'all_full_image', img_name)

                img = Image.open(img_path).convert('RGB')

                assert (img is not None) and (img.size > 0), \
                    ">> [ERROR] Image ( " + img_path + " ) is empty"

                p1x1 = int(line[1])
                p1y1 = int(line[2])
                p1x2 = int(line[3])
                p1y2 = int(line[4])

                p2x1 = int(line[5])
                p2y1 = int(line[6])
                p2x2 = int(line[7])
                p2y2 = int(line[8])

                pux1 = min(p1x1, p2x1)
                puy1 = min(p1y1, p2y1)
                pux2 = max(p1x2, p2x2)
                puy2 = max(p1y2, p2y2)

                context_crop = img.crop((pux1, puy1, pux2, puy2))

                assert (context_crop is not None) and (context_crop.size > 0), \
                    ">> [ERROR] Relation " + \
                    str(relation) + " context is empty"

                dir_name = os.path.join(
                    root_path, partition, label, 'relation_'+str(relation), 'context')

                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                    logger.log_dir()

                context_crop.save(dir_name+"/context.jpg")
                logger.log_patch(partition, int(label), 'context')

            print(">> ...done!")

            list.close()

        logger.save()


def main(args):

    if(args.dataset == 'PISC'):
        print(">> Processing " + args.dataset + "...")
        build_structure_PISC(args)
        build_PISC_relations_patches(args)
        build_PISC_bodies(args)
        build_PISC_objects(args)
        build_PISC_context(args)
        print(">> ... " + args.dataset + " finished!")

    if(args.dataset == 'PIPA'):
        print(">> Processing " + args.dataset + "...")
        build_structure_PIPA(args)
        build_PIPA_relations_patches(args)
        build_PIPA_bodies(args)
        build_PIPA_objects(args)
        build_PIPA_context(args)
        print(">> ... " + args.dataset + " finished!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description='PIPA and PISC data process for NU')
    parser.add_argument('--data_dir', metavar='DIR', type=dir_path,
                        help='path to dataset folder contaning splits, annotations and images')
    parser.add_argument('--save_dir', metavar='DIR',
                        type=dir_path, help='path to saving folder')
    parser.add_argument('--dataset', type=str, help='which dataset to process', choices=[
        'PIPA',
        'PISC'
    ])

    args = parser.parse_args()
    main(args)
