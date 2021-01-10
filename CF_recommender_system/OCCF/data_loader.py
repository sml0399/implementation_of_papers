import os
import os.path
import numpy as np
import random
import itertools as ite

# (original) dataset content: user id/ item id /rating /timestamp
# (data_loader.py) dataset content: user id/ item id /rating

# Usage Example:
# import data_loader as dl
# dataset=dl.loader_1M()
# splitted_dataset=dl.split_dataset( dataset )

total_100k_user=943
total_100k_item=1682
total_1M_user=6040
total_1M_item=3952


def dataset_to_matrix(dataset):
	user=list(set([i[0] for i in dataset]))
	item=list(set([i[1] for i in dataset]))
	rating_matrix=np.zeros((max(user),max(item)),dtype=np.intc)
	for data in dataset:
		rating_matrix[data[0]-1][data[1]-1]=int(data[2])
	return rating_matrix




def total_data_info(dataset_type="100k"):
	if(dataset_type=="100k"):
		return (total_100k_user,total_100k_item)
	if(dataset_type=="1M"):
		return (total_1M_user,total_1M_item)



def dataset_to_rating(dataset,data_type="100k"):
	rating_matrix=np.zeros(total_data_info(data_type),dtype=np.intc)
	for data in dataset:
		rating_matrix[data[0]-1][data[1]-1]=int(data[2])
	return rating_matrix



def rating_to_preference(rating):
	return np.array(np.vectorize(lambda x: 1 if x>0 else 0)(rating), dtype = np.float64)

def rating_to_confidence(rating,option=1):
	if(option==1):
		confidence_matrix=1+40*rating
	elif(option==2)
		confidence_matrix=1+40*np.log(1+rating/(10**(-8)))
	return confidence_matrix


def loader_1M():

	# load 1M dataset from the directory
	# no input
	# returns 2D ndarray type dataset

	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/1M_dataset/ratings.dat',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("::")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k():

	# load 100k dataset from the directory
	# no input
	# returns 2D ndarray type dataset

	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u.data',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text


def split_dataset(dataset, num_fold=5):

	# split the given dataset
	# dataset and number of folds(default 5) as input
	# returns 4D ndarray type dataset :   [ [trainset, testset] * num_fold ] with trainset and testset being 2D ndarray dataset
	
	if(num_fold<2 or num_fold==None):
		print("error in num_fold argument")
		return -1
	data_length=len(dataset)
	index_array=np.arange(data_length)
	random.shuffle(index_array)
	start=stop=0
	resulting_set=[]
	for i in range(num_fold):
		start = stop
		stop += len(index_array) // num_fold
		if i < len(index_array) % num_fold:
			stop += 1

		trainset = np.asarray([dataset[j] for j in ite.chain(index_array[:start],index_array[stop:])],dtype=object)
		testset = np.asarray([dataset[j] for j in index_array[start:stop]],dtype=object)
		resulting_set.append(np.asarray([trainset,testset],dtype=object))
	return np.asarray(resulting_set,dtype=object)
		

def get_dataset_info(dataset):
	
	# (return mean, # of users, # of items) of dataset
	# input: dataset of form [user id, item id, rating]
	user=list(set([i[0] for i in dataset]))
	item=list(set([i[1] for i in dataset]))
	num_user=len(user)
	num_item=len(item)
	data_mean=np.mean([i[2] for i in dataset])
	return (data_mean, num_user, num_item,user,item)

# train_set given by 100k
def loader_100k_1():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u1.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_2():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u2.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_3():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u3.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_4():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u4.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_5():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u5.base',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text



# test_set given by 100k
def loader_100k_t1(): 
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u1.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t2():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u2.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t3():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u3.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t4():
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u4.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text

def loader_100k_t5(): 
	f = open(os.path.dirname(__file__) + '/../../dataset/MovieLens/100k_dataset/u5.test',"r")
	text=f.read().split('\n')
	new_text=np.asarray([np.asarray([int(element) for element in row.split("\t")[0:3]]) for row in text[0:-1]])
	return new_text
