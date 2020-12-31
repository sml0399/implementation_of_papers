import os
import os.path
import numpy as np
# user id/ item id /rating /timestamp


def loader_1M():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/1M_dataset/ratings.dat',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("::")[0:3]]) for row in text[0:-1]])
	return new_text


# train_set
def loader_100k_1():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u1.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_2():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u2.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_3():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u3.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_4():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u4.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_5():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u5.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text


# test_set
def loader_100k_t1(): 
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u1.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t2():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u2.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t3():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u3.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t4():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u4.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t5(): 
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u5.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([np.float(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text
