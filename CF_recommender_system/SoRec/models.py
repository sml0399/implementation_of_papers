# SoRec
import numpy as np
import os
#from numba import vectorize, cuda
import data_loader as dl
import accuracy
import time
from sklearn.metrics import mean_absolute_error
from numpy import linalg as LA


class SoRec():
	def __init__(self,n_user,n_item,lambda_c=10,lambda_z=0.001,lambda_u=0.001,lambda_v=0.001,n_factor=10):
		super(SoRec,self).__init__()
		self.lambda_c = lambda_c
		self.lambda_z = lambda_z
		self.lambda_u = lambda_u
		self.lambda_v = lambda_v
		self.n_user=n_user
		self.n_item=n_item
		self.n_factor=n_factor
		self.U = np.random.normal(0, 0.1, (self.n_user,self.n_factors))
		self.V = np.random.normal(0, 0.1, (self.n_item,self.n_factors))
		self.Z = np.random.normal(0, 0.1, (self.n_user,self.n_factors))

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def sigmoid_prime(self,x):
		return np.exp(x)/(1+np.exp(x))**2

	def forward(self):  #return (prediction, loss)
		return (self.sigmoid(self.U@self.V.T),self.sigmoid(self.U@self.Z.T))

	def accuracy(self,rating_true):  # MAE 
		prediction=self.U@(self.V.T)
		return mean_absolute_error(rating_true,prediction)

	def update(self):
		U_prev=self.U
		V_prev=self.V
		Z_prev=self.Z
		

	def save_parameters(self, name="SVD.txt"):
		parameters=[]
		parameters.append(str(np.float64(self.mu)))						# mu
		parameters.append('\t'.join([str(b) for b in self.bu]))					# bu
		parameters.append('\t'.join([str(b) for b in self.bi]))					# bi
		(a,b)=np.shape(self.p)
		parameters.append(str(a)+'\t'+str(b))
		parameters.append('\t'.join([str(aa) for bb in self.p for aa in bb ]))			# p
		(c,d)=np.shape(self.q)
		parameters.append(str(c)+'\t'+str(d))
		parameters.append('\t'.join([str(aa) for bb in self.q for aa in bb ]))			# q
		(e,f)=np.shape(self.trainset)
		parameters.append(str(e)+'\t'+str(f))
		parameters.append('\t'.join([str(aa) for bb in self.trainset for aa in bb ]))		# trainset
		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/"+name,"w+")
		f.write('\n'.join(parameters))
		f.close()
		print("success : saving parameters")


	def load_parameters(self, name="SVD.txt"):

		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/"+name,"r")
		read_text=f.read()
		f.close()
		read_text=read_text.split('\n')

		self.mu=float(read_text[0])
		self.bu=np.asarray([np.float(a) for a in read_text[1].split('\t')])
		self.bi=np.asarray([np.float(b) for b in read_text[2].split('\t')])
		(a,b)=read_text[3].split('\t')
		(c,d)=read_text[5].split('\t')
		(e,f)=read_text[7].split('\t')
		self.p=np.asarray(read_text[4].split('\t'),dtype=np.float).reshape(int(a),int(b) )
		self.q=np.asarray(read_text[6].split('\t'),dtype=np.float).reshape(int(c),int(d) )
		self.q=np.asarray(read_text[8].split('\t'),dtype=np.float).reshape(int(e),int(f) )
		print("success : loading parameters")



