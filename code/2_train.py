# Training process for CNN model:
# CNN models spec is in /cnn
# By experiments, we run 90 epochs and might get good and stable accuracy

import numpy as np
import os
from cnn.cnn import Cnn
from utl import d_dir

x_tr = np.load(os.path.join(d_dir, "x_tr.npy"))
x_te = np.load(os.path.join(d_dir, "x_te.npy"))
y_tr = np.load(os.path.join(d_dir, "y_tr.npy"))
y_te = np.load(os.path.join(d_dir, "y_te.npy"))

cnn1 = Cnn()
tmp = cnn1.m.fit(x_tr, y_tr, validation_data=(x_te, y_te), batch_size=32, epochs=90, verbose=2)
cnn1.sv()

print(tmp.history["acc"])
print(tmp.history["val_acc"])
