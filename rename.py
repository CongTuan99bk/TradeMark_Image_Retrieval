# -*- coding: utf-8 -*-
"""Rename.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13qlveXt8OJH9nKej50SqjQt45FdIGfQk
"""

from google.colab import drive
drive.mount('/content/drive')

path0 = "/content/drive/MyDrive/DataAI_t/"
path1 = "/content/drive/MyDrive/DataAI_t/Compass/"
path2 = "/content/drive/MyDrive/DataAI_t/Human/"
path3 = "/content/drive/MyDrive/DataAI_t/Lion/"
path4 = "/content/drive/MyDrive/DataAI_t/Sun/"

import os
from PIL import Image as PImage
import cv2
from os import listdir
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import csv
import pandas

# Rename tên file
# path = "/content/drive/MyDrive/Data3/SUN/"
count = 0
# count được bắt đầu từ 0 và tăng dần theo các file ảnh
path = path2
imageL = listdir(path)
for img in imageL:
  scr = path + img
  des = path + "Human_" + str(count) + ".jpg"
  os.rename(scr, des)
  count+=1
print(count)