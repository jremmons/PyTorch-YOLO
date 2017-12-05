#!/usr/bin/env python3
import torch
import numpy as np
from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET

from multiprocessing import Pool
import os

from .model.tinyyolo import *


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


def load_class_set():
    return class_set


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

def load_cnn(weights, offset, conv2d, bn=None):
    if bn is not None:
        n_weights = conv2d.weight.numel()
        n_bias = bn.bias.numel()
        bn.bias.data.copy_(torch.from_numpy(weights[offset:offset+n_bias]))
        offset += n_bias
        bn.weight.data.copy_(torch.from_numpy(weights[offset:offset+n_bias]))
        offset += n_bias
        bn.running_mean.copy_(torch.from_numpy(weights[offset:offset+n_bias]))
        offset += n_bias
        bn.running_var.copy_(torch.from_numpy(weights[offset:offset+n_bias]))
        offset += n_bias
        conv2d.weight.data.copy_(torch.from_numpy(weights[offset:offset+n_weights]).view_as(conv2d.weight.data))
        offset += n_weights
    else:
        n_weights = conv2d.weight.numel()
        n_bias = conv2d.bias.numel()
        conv2d.bias.data.copy_(torch.from_numpy(weights[offset:offset+n_bias]))
        offset += n_bias
        conv2d.weight.data.copy_(torch.from_numpy(weights[offset:offset+n_weights]).view_as(conv2d.weight.data))
        offset += n_weights

    return offset


def load_weights(model, path):
    weights = np.fromfile(path, dtype=np.float32)

    offset = 4
    offset = load_cnn(weights, offset, conv2d=model[0],  bn=model[1])
    offset = load_cnn(weights, offset, conv2d=model[4],  bn=model[5])
    offset = load_cnn(weights, offset, conv2d=model[8],  bn=model[9])
    offset = load_cnn(weights, offset, conv2d=model[12], bn=model[13])
    offset = load_cnn(weights, offset, conv2d=model[16], bn=model[17])
    offset = load_cnn(weights, offset, conv2d=model[20], bn=model[21])
    offset = load_cnn(weights, offset, conv2d=model[24], bn=model[25])
    offset = load_cnn(weights, offset, conv2d=model[27], bn=model[28])
    offset = load_cnn(weights, offset, conv2d=model[30])

    print ('Weights loading done.')

def preprocess(im):
    width, height = im.size
    im = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
    im = im.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    im = im.view(1, 3, height, width)
    im = im.float().div(255.0)
    return im


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def get_color(indice, classes):
    r = indice * 3141 % 255
    g = indice * 5926 % 255
    b = indice * 5358 % 255
    return (r, g, b)


def draw_box(img, boxes, color, class_names=None, save_name=None):
    '''
    Data format in Bounding boxes:
        [0, 1, 2, 3]: Center X, Center Y, Length X, Length Y
        [4]: Trust
        [5]: Class Index
    '''
    w = img.width
    h = img.height
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x1 = (box[0] - box[2] / 2.0) * w
        y1 = (box[1] - box[3] / 2.0) * h
        x2 = (box[0] + box[2] / 2.0) * w
        y2 = (box[1] + box[3] / 2.0) * h

        rgb = [255, 0, 0]
        if class_names != None:
            classes = len(class_names)
            class_id = int(box[6])  # if class_id is box[5]
            rgb = get_color(class_id, classes)
            draw.text((x1, y1), class_names[class_id], fill=rgb)

        line = (x1,y1,x1,y2)
        draw.line(line, fill=rgb, width=5)
        line = (x1,y1,x2,y1)
        draw.line(line, fill=rgb, width=5)
        line = (x1,y2,x2,y2)
        draw.line(line, fill=rgb, width=5)
        line = (x2,y1,x2,y2)
        draw.line(line, fill=rgb, width=5)

    if save_name:
        img.save(save_name)

    return img

def demo(tiny_yolo, mp4_file):
    class_names = load_class_names('../../data/voc.names')

    cap = cv2.VideoCapture(mp4_file)
    if not cap.isOpened():
        print("Unable to open the mp4 file.")
        return

    out = cv2.VideoWriter("output2.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    i = 0
    while True:
        i += 1
        res, img = cap.read()
        # if i % 3 == 0:
        #     continue
        if res:
            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)

            pil_im2 = preprocess(pil_im)
            boxes = tiny_yolo.detect(pil_im2)

            draw_img = draw_box(pil_im, boxes, None, class_names)

            cv2_im2 = cv2.cvtColor(np.array(draw_img), cv2.COLOR_RGB2BGR)
            print (cv2_im2.shape)

            out.write(cv2_im2)


    cv2.destroyAllWindows()
    out.release()
    cap.release()

    print ('done.')

def get_color(indice, classes):
    r = indice * (classes - 1) % 255
    g = indice * (classes) % 255
    b = indice * (classes + 1) % 255
    return (r, g, b)


def draw_box(img, boxes, color, class_names=None, save_name=None):
    '''
    Data format in Bounding boxes:
        [0, 1, 2, 3]: Center X, Center Y, Length X, Length Y
        [4]: Trust
        [5]: Class Index
    '''
    w = img.width
    h = img.height
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x1 = (box[0] - box[2] / 2.0) * w
        y1 = (box[1] - box[3] / 2.0) * h
        x2 = (box[0] + box[2] / 2.0) * w
        y2 = (box[1] + box[3] / 2.0) * h

        rgb = [255, 0, 0]
        if class_names != None:
            classes = len(class_names)
            class_id = box[5]  # if class_id is box[5]
            rgb = get_color(class_id, classes)
            draw.text((x1, y1), class_names[class_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)

    if save_name:
        img.save(save_name)

    return img


def test():
    train_dict, valid_dict = get_train_valid_dict()
    print("Train data dictionary:")
    print(train_dict)

    print("Valid data dictionary:")
    print(valid_dict)


if __name__ == "__main__":
    test()
