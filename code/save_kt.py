# We save indexed k-d tree as files to be used for the following times

import _pickle as cPickle
import os
from utl import d_dir
from knn import Knn

knn1 = Knn()
for i, v in knn1.kt.items():
	with open(os.path.join(d_dir, "kt_"+str(i)+".pickle"), 'wb') as f:
		cPickle.dump(v, f)
