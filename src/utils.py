#!/usr/bin/env python

from PIL import Image

import xml.etree.ElementTree as ET

from multiprocessing import Pool
import os

data_sets = [("2007", "train"), ("2007", "val")]

class_set = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

DATA_BASE_DIR = "../data/train"
TRAINING_IMAGE_DIR = "train_img/"


def generate_object_dict(class_index, bndbox, w, h):
    ret = {'label': class_index}
    xcenter = (bndbox[0] + bndbox[1]) // 2
    ycenter = (bndbox[2] + bndbox[3]) // 2
    xlength = (bndbox[1] - bndbox[0])
    ylength = (bndbox[3] - bndbox[2])
    ret['xcenter'] = xcenter / w
    ret['ycenter'] = ycenter / h
    ret['xlength'] = xlength / w
    ret['ylength'] = ylength / h
    return ret


def get_img_object_list(year, img_name):
    img_xml = "VOC%s/Annotations/%s.xml" % (year, img_name)
    with open(os.path.join(DATA_BASE_DIR, img_xml)) as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

    objects = []
    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        name = obj.find("name").text
        if name not in class_set or int(difficult) == 1:
            # print("Image has the class %s we don't want to classify "
            # "or the difficulty = %s is not fit." % (name, difficult))
            continue
        class_index = class_set.index(name)
        bndbox = obj.find("bndbox")
        bndbox = (float(bndbox.find('xmin').text),
                  float(bndbox.find('xmax').text),
                  float(bndbox.find('ymin').text),
                  float(bndbox.find('ymax').text))
        objects.append(generate_object_dict(class_index, bndbox, w, h))
    return objects


def convert_img_to_training_size(year_img_name_tuple):
    year = year_img_name_tuple[0]
    img_name = year_img_name_tuple[1]

    img_path = "VOC%s/JPEGImages/%s.jpg" % (year, img_name)
    img = Image.open(os.path.join(DATA_BASE_DIR, img_path))
    img = img.resize((448, 448), Image.ANTIALIAS)
    img.save(os.path.join(TRAINING_IMAGE_DIR, "%s.jpg" % img_name))


def convert_data_to_dict(year, data_file_suffix):
    data_dict = {}
    if (year, data_file_suffix) not in data_sets:
        print("No an available data sets.")
        return data_dict

    img_filenames = get_img_file_set(year, data_file_suffix)

    if not os.path.exists(TRAINING_IMAGE_DIR):
        os.makedirs(TRAINING_IMAGE_DIR)

    with Pool(4) as p:
        p.map(convert_img_to_training_size,
              map(lambda n: (year, n), img_filenames))
    for img_name in img_filenames:
        data_dict["%s.jpg" % img_name] = get_img_object_list(year, img_name)
    return data_dict


def get_train_valid_dict():
    train_dict = convert_data_to_dict("2007", "train")
    valid_dict = convert_data_to_dict("2007", "val")
    return train_dict, valid_dict


def get_img_file_set(year, suffix):
    catalog_file = "VOC%s/ImageSets/Main/%s.txt" % (year, suffix)
    with open(os.path.join(DATA_BASE_DIR, catalog_file)) as f:
        return f.read().strip().split()


def main():
    train_dict, valid_dict = get_train_valid_dict()
    print("Train data dictionary:")
    print(train_dict)

    # print("Valid data dictionary:")
    # print(valid_dict)


if __name__ == "__main__":
    main()
