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
num_items=61035
num_users=1000

def info():
	return (num_users,num_items)

def load_rating():
	rating_txt=open("/../../dataset/epinion/epinion/ratings_data.txt", "r").read()
	temp=rating_txt.split('\n')
	temp[0]=temp[0][-7:]
	rating_raw=[[int(b[0]),int(b[1]),int(b[2])] for b in [a.split(' ') for a in temp[:-1]]]
	rating_item_new_index=[]
	rating_reduced=[]
	for user_id, item_id, rating_val in rating_raw:
	    if(user_id>1000):
		break
	    rating_item_new_index.append(item_id)
	rating_item_new_index=list(set(rating_item_new_index))
	for user_id, item_id, rating_val in rating_raw:    
	    if(user_id>1000):
		break
	    rating_reduced.append([(user_id-1),rating_item_new_index.index(item_id),rating_val])
	ratings=np.zeros((num_users,num_items),dtype=np.float64)
	for user_id, item_id, rating_val in rating_reduced:
		ratings[user_id][item_id]=(rating_val-1)/4
	return ratings

def load_trust():
	trust_txt=open("/../../dataset/epinion/epinion/trust_data.txt", "r").read()
	temp=trust_txt.split('\n')
	temp[0]=temp[0][-14:]
	trust_raw=[[int(b[0]),int(b[1])] for b in [a.split(' ')[1:-1] for a in temp[:-1]]]
	trust_reduced=[[a[0]-1,a[1]-1] for a in trust_raw if (a[0]<=1000 and a[1]<=1000)]
	trusts=np.zeros((num_users,num_users),dtype=np.float64)
	for user_1, user_2 in trust_reduced:
		trusts[user_1][user_2]=1
	return trusts

def load_confidence(trusts):
	confidence=np.zeros((num_users,num_users),dtype=np.float64)
	for user_1, user_2 in trust_reduced:
		in_degree=np.count_nonzero(trusts[:,user_2])
		out_degree=np.count_nonzero(trusts[user_1,:])
		confidence[user_1][user_2]=(in_degree/(in_degree+out_degree))**0.5*trusts[user_1,user_2]
	return confidence

def load_IR(ratings):
	IR=np.copy(ratings)
	return IR[IR>0]=1

def load_IC(confidence):
	IC=np.copy(confidence)
	return IC[IC>0]=1



