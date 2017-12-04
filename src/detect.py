#!/usr/bin/env python3

from PIL import ImageDraw
'''
Data format in Bounding boxes:
    [0, 1, 2, 3]: Center X, Center Y, Length X, Length Y
    [4]: Trust
    [5]: Class Index
'''


def get_color(indice, classes):
    r = indice * (classes - 1) % 255
    g = indice * (classes) % 255
    b = indice * (classes + 1) % 255
    return (r, g, b)


def draw_box(img, boxes, color, class_names=None, save_name=None):
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
