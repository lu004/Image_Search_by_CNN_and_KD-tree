# global file paths for images, vectors of images, k-d tree of features of images

import json
import os

img_dir = os.path.join(os.getcwd(), "geological_similarity/geological_similarity")
d_dir = os.path.join(os.getcwd(), "d")

tmp = os.path.join(d_dir, "img_f.json")
if os.path.exists(tmp):
	img_f = json.load(open(tmp, "r"))

tmp = os.path.join(d_dir, "lab.json")
if os.path.exists(tmp):
	lab = json.load(open(tmp, "r"))