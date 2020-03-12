# We run an experiment for running time of search.
# We compare the running time of search by k-d tree and the time of search by brute-force search

import numpy as np
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utl import d_dir, img_dir, img_f, lab
from keras.applications.mobilenet_v2 import preprocess_input
from knn import Knn

x_all = np.load(os.path.join(d_dir, "x.npy"))
x_all = preprocess_input(x_all)
x = np.load(os.path.join(d_dir, "x_te.npy"))
top = 50

# knn time
knn = Knn()
x = x[:1000]
re = []
for k in list(range(1, top+1)):
	print(k)
	s = time.time()
	for i in x:
		knn.run_top(i, k)
	re.append((time.time() - s)/len(x))

# brute-force time
x = x[:10]
re2 = []
for k in list(range(1, top+1)):
	s = time.time()
	for i in x:
		sc = []
		for z in x_all:
			sc.append(np.linalg.norm(i-z))
		sorted(sc)[:k]
	re2.append((time.time() - s)/len(x))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("top k of similar images", size=14)
ax.set_ylabel("run time per query (second)", size=14)
for label, v in {"search by k-d tree (we used)": re, "search by brute-force": re2}.items():
	ax.plot(list(range(1, len(v)+1)), v, linestyle="solid", marker='o', label=label)
	for i, j in zip(list(range(1, len(v)+1)), v):
		if i % 10 ==0:
			ax.annotate("{:.4f}".format(j), xy=(i, j), size=10)
ax.legend()