# Running the search system.
# note that we should specify the file path and the top-k of outputs

import argparse
import os
import cv2
from knn import Knn
from utl import img_dir, img_f

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_path", type=str, default=os.path.join(img_dir, img_f[str(17000)]), help="query image's file path")
parser.add_argument("-k", "--top_k", type=int, default=10, help='top k of similar images')
ag = parser.parse_args()
pth = ag.img_path
k = ag.top_k

knn = Knn()
while True:
	if os.path.exists(pth):
		q, top, top_pth = knn.get_topk(cv2.imread(pth, 1), k)
		print("query image: {}".format(pth))
		print("similar images (top {}): {}".format(k, top_pth))
		cv2.imshow("query", q)
		cv2.imshow("similar images (top {}), press a key to close".format(k), top)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print("wrong file path")
	k = int(input("input top k: ") or "10")
	pth = input("input file path of query image: ")
	pth = pth.replace('\'', '')
