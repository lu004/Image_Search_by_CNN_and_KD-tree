# We visualize the CNN feature values of all 30,000 images
# by projecting them into a 2D plane

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.applications.mobilenet_v2 import preprocess_input
from cnn.cnn import Cnn
from utl import d_dir

cmap = get_cmap("tab10")

x = np.load(os.path.join(d_dir, "x.npy"))
y = np.load(os.path.join(d_dir, "y.npy"))
x = preprocess_input(x)
yc = [cmap(np.argmax(i)) for i in y]

# x = x[:10000]
# yc = yc[:10000]

cnn1 = Cnn()
x1 = cnn1.get_f(x)
#x2 = PCA(n_components=0.8).fit_transform(x1)
#x3 = TSNE(n_components=2, perplexity=40, learning_rate=400, n_iter=500).fit_transform(x2)
x3 = TSNE(n_components=2).fit_transform(x1)

plt.scatter(x3[:, 0], x3[:, 1], c=yc, marker=".", alpha=0.2)
plt.legend()
plt.show()