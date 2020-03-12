# Data preprocessing for Geological image data
# - normalization for rgb values of images
# - build dictionary of file path of image

import numpy as np
import os
import cv2
import json
from utl import d_dir, img_dir
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet_v2 import preprocess_input

lab = {v: i for i, v in enumerate(os.listdir(img_dir))}
img_f = []

x = [] # img
y = [] # label
for f in os.listdir(img_dir):
    for f2 in os.listdir(os.path.join(img_dir, f)):
        tmp = cv2.imread(os.path.join(img_dir, f, f2), 1)
        x.append(tmp)
        img_f.append(f+"/"+f2)
        tmp = np.zeros(len(lab))
        tmp[lab[f]] = 1
        y.append(tmp)
x = np.array(x)
y = np.array(y)
img_f = {i: v for i, v in (img_f)}

np.save(os.path.join(d_dir, "x"), x)
np.save(os.path.join(d_dir, "y"), y)
json.dump(lab, open(os.path.join(d_dir, "lab.json"), 'w'))
json.dump(img_f, open(os.path.join(d_dir, "img_f.json"), 'w'))


x = preprocess_input(x)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3)
np.save(os.path.join(d_dir, "x_tr"), x_tr)
np.save(os.path.join(d_dir, "x_te"), x_te)
np.save(os.path.join(d_dir, "y_tr"), y_tr)
np.save(os.path.join(d_dir, "y_te"), y_te)
