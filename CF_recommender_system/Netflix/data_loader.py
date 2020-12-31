import os
import os.path
import numpy as np
# user id/ item id /rating /timestamp


def loader_100k():
	f = open(os.path.dirname(__file__) + '../../dataset/MovieLens/1M_dataset/ratings.dat',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("::")[0:3]]) for row in text[0:-1]])
	print(new_text[0:2])


def loader_1M():
	f = open(os.path.dirname(__file__) + '../../dataset/MovieLens/100k_dataset/u1.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	print(new_text[0:2])


loader_100k()
loader_1M()
