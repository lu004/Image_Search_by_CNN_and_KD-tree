# We use k-d tree to efficiently search for images.
# We need to build the tree for the first time
# and we store the indexed-tree as files and use the tree for the following times

import scipy.spatial as ss
import _pickle as cPickle
import numpy as np
import os
import cv2
from keras.applications.mobilenet_v2 import preprocess_input
from utl import d_dir, img_dir, img_f, lab
from cnn.cnn import Cnn

class Knn:

	# def __init__(self):
	# 	x = np.load(os.path.join(d_dir, "x.npy"))
	# 	y = np.load(os.path.join(d_dir, "y.npy"))
	# 	x = preprocess_input(x)
	# 	self.cnn = Cnn()
	# 	self.kt = {}
	# 	for v in lab.values():
	# 		x1 = x[y[:, v] == 1]
	# 		x1 = self.cnn.get_f(x1)
	# 		self.kt[v] = ss.KDTree(x1, leafsize=10)

	def __init__(self):
		# x = np.load(os.path.join(d_dir, "x.npy"))
		# y = np.load(os.path.join(d_dir, "y.npy"))
		# x = preprocess_input(x)
		self.cnn = Cnn()
		self.kt = {}
		# for v in lab.values():
		# 	x1 = x[y[:, v] == 1]
		# 	x1 = self.cnn.get_f(x1)
		# 	self.kt[v] = ss.KDTree(x1, leafsize=10)
		for i in range(6):
			with open(os.path.join(d_dir, "kt_" + str(i) + ".pickle"), 'rb') as f:
				self.kt[i] = cPickle.load(f)

	def get_topk(self, q, k):
		q = cv2.resize(q, (28, 28))
		q2 = preprocess_input(np.expand_dims(q, axis=0))
		cl = np.argmax(self.cnn.pred(q2)[0])
		f = self.cnn.get_f(q2)[0]

		pth = []
		for i in self.kt[cl].query(f, k)[1]:
			pth.append(img_f[str(cl*5000+i)])
		# show
		size = 4
		top = np.array([])
		for i, f in enumerate(pth):
			img = cv2.resize(cv2.imread(os.path.join(img_dir, f), 1), None, fx=size, fy=size)
			top = np.concatenate((top, img), axis=1) if top.size else img
		return q, top, pth

	def run_top(self, q, k):
		q = np.expand_dims(q, axis=0)
		cl = np.argmax(self.cnn.pred(q)[0])
		f = self.cnn.get_f(q)[0]
		self.kt[cl].query(f, k)
		return None
