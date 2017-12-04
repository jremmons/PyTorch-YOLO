#!/usr/bin/env python3

import sys

import cv2

from model.tiny_yolo import TinyYoloNet
from utils import load_cnn, load_weights, load_class_set, draw_box


def demo(mp4_file):
    n = TinyYoloNet()
    load_cnn()
    load_weights()

    class_set = load_class_set()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("out.mp4", fourcc, 20.0, (n.width, n.height))

    cap = cv2.VideoCapture(mp4_file)
    if not cap.isOpened():
        print("Unable to open the mp4 file.")
        return

    while True:
        res, img = cap.read()
        if res:
            resized_img = cv2.resize(img, (n.width, n.height))
            bboxes = n.forward(resized_img, 0.5, 0.4)
            draw_img = draw_box(resized_img, bboxes, None, class_set)

            out.write(draw_img)

            cv2.imshow("demo", draw_img)

            if cv2.waitKey(1) == 27:
                break
        else:
            print("Unable to read image.")
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        mp4_file = sys.argv[1]
        demo(mp4_file)
    else:
        print('Usage:')
        print('    python demo.py mp4_file')
        print('')
        print('    perform detection on MP4 video file')
