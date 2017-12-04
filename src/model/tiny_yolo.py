import torch
import torch.nn as nn

from src.utils import load_weights

from collections import OrderedDict

N_CLASSES = 20
ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
N_ANCHORS = len(ANCHORS) / 2
N_OUTPUT = int(N_ANCHORS * (5 + N_CLASSES))


class TinyYoloNet(nn.Module):
    def __init__(self, path=''):
        super(TinyYoloNet, self).__init__()

        self.model = nn.Sequential(
            OrderedDict([
                # 1
                ("conv1", nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
                 ),
                ("bn1", nn.BatchNorm2d(16)),
                ("leaky1", nn.LeakyReLU(0.1, inplace=True)),
                ("pool1", nn.MaxPool2d(2, 2)),

                # 2
                ("conv2", nn.Conv2d(
                    16, 32, 3, stride=1, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(32)),
                ("leaky2", nn.LeakyReLU(0.1, inplace=True)),
                ("pool2", nn.MaxPool2d(2, stride=2)),

                # 3
                ("conv3", nn.Conv2d(
                    32, 64, 3, stride=1, padding=1, bias=False)),
                ("bn3", nn.BatchNorm2d(64)),
                ("leaky3", nn.LeakyReLU(0.1, inplace=True)),
                ("pool3", nn.MaxPool2d(2, stride=2)),

                # 4
                ("conv4", nn.Conv2d(
                    64, 128, 3, stride=1, padding=1, bias=False)),
                ("bn4", nn.BatchNorm2d(128)),
                ("leaky4", nn.LeakyReLU(0.1, inplace=True)),
                ("pool4", nn.MaxPool2d(2, stride=2)),

                # 5
                ("conv5", nn.Conv2d(
                    128, 256, 3, stride=1, padding=1, bias=False)),
                ("bn5", nn.BatchNorm2d(256)),
                ("leaky5", nn.LeakyReLU(0.1, inplace=True)),
                ("pool5", nn.MaxPool2d(2, stride=2)),

                # 6
                ("conv6", nn.Conv2d(
                    256, 512, 3, stride=1, padding=1, bias=False)),
                ("bn6", nn.BatchNorm2d(512)),
                ("leaky6", nn.LeakyReLU(0.1, inplace=True)),
                ("pool6", nn.MaxPool2d(2, stride=1)),

                # 7
                ("conv7", nn.Conv2d(
                    512, 1024, 3, stride=1, padding=1, bias=False)),
                ("bn7", nn.BatchNorm2d(1024)),
                ("leaky7", nn.LeakyReLU(0.1, inplace=True)),

                # 8
                ("conv8", nn.Conv2d(
                    1024, 1024, 3, stride=1, padding=1, bias=False)),
                ("bn8", nn.BatchNorm2d(1024)),
                ("leaky8", nn.LeakyReLU(0.1, inplace=True)),

                # 9
                ("conv9", nn.Conv2d(1024, N_OUTPUT, 1, stride=1, padding=0)),
            ]))

        if len(path) != 0:
            print('Loading weights...')
            load_weights(self.model, path)

            #weights = np.fromfile(path, dtype=np.float32)

    def forward(self, x):
        y = self.model(x)
        return y

    def nms(self, boxes, nms_thresh):
        if len(boxes) == 0:
            return boxes

        rem_confs = torch.zeros(len(boxes))
        for i in range(len(boxes)):
            '''
            boxes[4] is the confidence for the this object label.
            We use rem_confs to have an ascending order for confidence.
            '''
            rem_confs[i] = 1 - boxes[i][4]

        len_boxes = len(boxes)
        _, sort_index = torch.sort(rem_confs)
        ret = []
        for i in range(len_boxes):
            box_i = boxes[sort_index[i]]
            if box_i[4] > 0:
                ret.append(box_i)
                # Eliminate the following boxes with iou over threshold.
                for j in range(i + 1, len_boxes):
                    box_j = boxes[sort_index[j]]
                    if self.bbox_iou(box_i, box_j) > nms_thresh:
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
        return shared_area / outer_area


tiny_yolo = TinyYoloNet('../../tiny-yolo-voc.weights')

# for module in tiny_yolo.modules():
#     print (module)
#
# print(tiny_yolo.state_dict().keys())
#
# print (tiny_yolo.state_dict())
# print (tiny_yolo.state_dict()['conv1.weight'])
# # print (tiny_yolo.parameters())

#print(tiny_yolo.state_dict().keys())
#print(tiny_yolo.model.state_dict().keys())
#print (tiny_yolo.model[0])
