import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import math
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.utils import *
from collections import OrderedDict

# 2 x 2 max pooling preserve dimension by padding 0.5
class MaxPool2d_(nn.Module):
    def __init__(self):
        super(MaxPool2d_, self).__init__()

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode='replicate')
        y = F.max_pool2d(x, 2, stride=1)
        return y

class TinyYoloNet(nn.Module):
    def __init__(self, path=''):
        super(TinyYoloNet, self).__init__()

        self.n_classes = 20
        self.anchors = [1.08, 1.19,  3.42, 4.41,  6.63, 11.38,  9.42, 5.11,  16.62, 10.52]
        self.n_anchors = int(len(self.anchors) / 2)
        n_outputs = int(self.n_anchors * (5 + self.n_classes))

        self.model = nn.Sequential(
            OrderedDict([
                # 1
                ("conv1",  nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)),
                ("bn1",    nn.BatchNorm2d(16)),
                ("leaky1", nn.LeakyReLU(0.1, inplace=True)),
                ("pool1",  nn.MaxPool2d(2, 2)),

                # 2
                ("conv2",  nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False)),
                ("bn2",    nn.BatchNorm2d(32)),
                ("leaky2", nn.LeakyReLU(0.1, inplace=True)),
                ("pool2",  nn.MaxPool2d(2, stride=2)),

                # 3
                ("conv3",  nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)),
                ("bn3",    nn.BatchNorm2d(64)),
                ("leaky3", nn.LeakyReLU(0.1, inplace=True)),
                ("pool3",  nn.MaxPool2d(2, stride=2)),

                # 4
                ("conv4",  nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)),
                ("bn4",    nn.BatchNorm2d(128)),
                ("leaky4", nn.LeakyReLU(0.1, inplace=True)),
                ("pool4",  nn.MaxPool2d(2, stride=2)),

                # 5
                ("conv5",  nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)),
                ("bn5",    nn.BatchNorm2d(256)),
                ("leaky5", nn.LeakyReLU(0.1, inplace=True)),
                ("pool5",  nn.MaxPool2d(2, stride=2)),

                # 6
                ("conv6",  nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)),
                ("bn6",    nn.BatchNorm2d(512)),
                ("leaky6", nn.LeakyReLU(0.1, inplace=True)),
                ("pool6",  MaxPool2d_()),

                # 7
                ("conv7",  nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False)),
                ("bn7",    nn.BatchNorm2d(1024)),
                ("leaky7", nn.LeakyReLU(0.1, inplace=True)),

                # 8
                ("conv8",  nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False)),
                ("bn8",    nn.BatchNorm2d(1024)),
                ("leaky8", nn.LeakyReLU(0.1, inplace=True)),

                # 9
                ("conv9",  nn.Conv2d(1024, n_outputs, 1, stride=1, padding=0)),
            ])
        )

        if len(path) != 0:
            print ('Loading weights...')
            load_weights(self.model, path)


    def forward(self, x):
        y = self.model(x)
        return y


    def nms(self, boxes, nms_thresh):
        class_names = load_class_names('../../data/voc.names')
        if len(boxes) == 0:
            return boxes

        rem_confs = torch.zeros(len(boxes))
        for i in range(len(boxes)):
            '''
            boxes[4] is the confidence for the this object label.
            We use rem_confs to have an ascending order for confidence.
            '''
            rem_confs[i] = 1 - boxes[i][4]
            print (class_names[boxes[i][6]])

        len_boxes = len(boxes)
        _, sort_index = torch.sort(rem_confs)
        ret = []
        for i in range(len_boxes):
            box_i = boxes[sort_index[i]]
            label_i = box_i[6]
            if box_i[4] > 0:
                ret.append(box_i)
                # Eliminate the following boxes with iou over threshold.
                for j in range(i + 1, len_boxes):
                    box_j = boxes[sort_index[j]]
                    label_j = box_j[6]
                    if label_i == label_j and self.bbox_iou(box_i, box_j) > nms_thresh:
                        box_j[4] = 0
        return ret


    def bbox_iou(self, box1, box2):
        # Get the most outer space for two boxes.
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        outer_w = Mx - mx
        outer_h = My - my
        shared_w = w1 + w2 - outer_w
        shared_h = h1 + h2 - outer_h
        if shared_w <= 0 or shared_h <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        shared_area = shared_w * shared_h
        outer_area = area1 + area2 - shared_area
        return max(max(shared_area/area1, shared_area/area1), shared_area / outer_area)
        #return

    def get_all_boxes(self, output, prob_thresh):
        batch_size, depth, height, width = output.size()

        ''' 1 125 13 13 -> 25 5*13*13 '''
        output = output.view(batch_size*self.n_anchors, 5+self.n_classes, height*width)\
                       .transpose(0, 1).contiguous()\
                       .view(5+self.n_classes, batch_size*self.n_anchors*height*width)

        ''' 5 13 13-> 5*13*13 '''
        grid_x = torch.linspace(0, width-1, width).repeat(height, 1)\
                      .repeat(batch_size*self.n_anchors, 1, 1)\
                      .view(batch_size*self.n_anchors*height*width)

        grid_y = torch.linspace(0, height-1, height).repeat(width, 1).t()\
                      .repeat(batch_size*self.n_anchors, 1, 1)\
                      .view(batch_size*self.n_anchors*height*width)

        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y

        # dim 1, col operation
        anchor_w = torch.Tensor(self.anchors).view(self.n_anchors, 2)\
                        .index_select(1, torch.LongTensor([0]))\
                        .repeat(batch_size, 1)\
                        .repeat(1, 1, height*width)\
                        .view(batch_size*self.n_anchors*height*width)

        anchor_h = torch.Tensor(self.anchors).view(self.n_anchors, 2)\
                        .index_select(1, torch.LongTensor([1]))\
                        .repeat(batch_size, 1)\
                        .repeat(1, 1, height*width)\
                        .view(batch_size*self.n_anchors*height*width)

        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3]) * anchor_h

        probs_obj = torch.sigmoid(output[4])

        out_classes = output[5:5+self.n_classes].transpose(0, 1)

        prob_classes = torch.nn.Softmax()(Variable(out_classes)).data

        # dim 1, row operation
        prob_max, max_idx = torch.max(prob_classes, 1)
        prob_max, max_idx = prob_max.view(-1), max_idx.view(-1)

        anchor_step = height * width
        batch_step  = anchor_step * self.n_anchors

        all_boxes = []
        for batch_id in range(batch_size):
            for k in range(self.n_anchors):
                for y in range(height):
                    for x in range(width):
                        idx = batch_id*batch_step + k * anchor_step + y * width + x
                        prob = probs_obj[idx]

                        if prob > prob_thresh:
                            box = [
                                xs[idx] / width,
                                ys[idx] / height,
                                ws[idx] / width,
                                hs[idx] / height,
                                prob,
                                prob_max[idx],
                                max_idx[idx]
                            ]
                            all_boxes.append(box)

        return all_boxes

    def detect(self, im):
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

        if torch.cuda.is_available():
            im = Variable(im).cuda()
        else:
            im = Variable(im)

        output = self.model(im)
        boxes  = self.get_all_boxes(output.data, 0.3)
        boxes  = self.nms(boxes, 0.4)

        return boxes


if __name__ == '__main__':

    # ## init
    # im_path = '../../img/'
    # #im = Image.open(im_path).convert('RGB').resize((416, 416))
    # #im = preprocess(im)
    # class_names = load_class_names('../../data/voc.names')
    # tiny_yolo = TinyYoloNet('../../tiny-yolo-voc.weights')
    #
    # ## inference
    # pattern = os.path.join(im_path, '*.jpg')
    # for i, filepath in enumerate(glob.glob(pattern), 1):
    #     if i > 100:
    #         break
    #     file_name = os.path.basename(filepath)
    #     im = Image.open(filepath).convert('RGB').resize((416, 416))
    #     im = preprocess(im)
    #     boxes = tiny_yolo.detect(im)
    #
    #     ## plot
    #     img = Image.open(filepath).convert('RGB')
    #     plot_boxes(img, boxes, file_name, class_names)

    tiny_yolo = TinyYoloNet('../../tiny-yolo-voc.weights')
    demo(tiny_yolo, '../../data/videoplayback2.mp4')
