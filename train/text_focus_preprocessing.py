#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def count_area_pixel(img, start_pixel, img_high):
    pixel_cnt = 0
    for i in range(start_pixel, start_pixel + 64):
        for j in range(img_high):
            if img[j][i] == 0:
                pixel_cnt += 1
    return pixel_cnt


def search_text_position(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    H, W = thresh.shape
    max_pixel_area = 0
    max_pixel_pos = 0

    for i in range(W - 64):
        tmp = count_area_pixel(thresh, i, H)
        if tmp >= max_pixel_area:
            max_pixel_area = tmp
            max_pixel_pos = i

    img = img[0:H, max_pixel_pos:max_pixel_pos + 64]

    c_H, c_W, channel = img.shape
    top = round(c_H / 7)
    bot = round(c_H / 7 * 6)
    img_top = img[0:top, 0:c_W]
    img_bot = img[bot:c_H, 0:c_W]
    is_line_threshold = 10
    minLineLength = 20
    maxLineGap = 10

    edges = cv2.Canny(img_top, 50, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, is_line_threshold, minLineLength, maxLineGap)

    try:
        lines = lines[:, 0, :]
        print(len(lines))
        for x1, y1, x2, y2 in lines:
            cv2.line(img_top, (x1, y1), (x2, y2), (255, 255, 255), 3)
    except Exception as e:
        print("This top img didn't detect any line")

    edges = cv2.Canny(img_bot, 50, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, is_line_threshold, minLineLength, maxLineGap)

    try:
        lines = lines[:, 0, :]
        print(len(lines))
        for x1, y1, x2, y2 in lines:
            cv2.line(img_bot, (x1, y1), (x2, y2), (255, 255, 255), 3)
    except Exception as e:
        print("This bottom img didn't detect any line")

    return img


try:
    file_name = sys.argv[1]
except IndexError:
    file_name = '5.jpg'

if file_name == '-h':
    print("usage: python text_focus_preprocessing.py [file] \t\t\t>>>looking for image in the path "
              "of datasets/raw_data/test_images/")
    print("example:\npython text_focus_preprocessing.py")
    print("python text_focus_preprocessing.py a.jpg")
    exit()


img = cv2.imread('datasets/raw_data/test_images/' + file_name)

output = search_text_position(img)
plt.imshow(output)
plt.show()


