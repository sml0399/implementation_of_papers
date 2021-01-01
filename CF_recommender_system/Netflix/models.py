# SVD
import numpy as np
import os


class SVD():
	def __init__(self,num_factors=100,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1	):
		self.trainset=0
		self.mu=0 #global mean mu
		self.bu=0
		self.bi=0
		self.p=0
		self.q=0

	def fit(self, train_set):
		self.trainset=train_set


	
	#def predict(dataset):
		



	def save_parameters(self):
		print(os.path.dirname(os.path.dirname(__file__)))

		parameters=[]
		parameters.append(str(np.float(self.mu)))
		parameters.append(str(np.float(self.bu)))
		parameters.append(str(np.float(self.bi)))
		parameters.append(str(np.float(self.p)))
		parameters.append(str(np.float(self.q)))
		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVD.txt","w+")
		f.write('\n'.join(parameters))
		f.close()
		print("success : saving parameters")

	def load_parameters(self):
		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVD.txt","r")
		read_text=f.read()
		f.close()
		read_text=[float(content) for content in read_text.split('\n')]
		self.mu=read_text[0]
		self.bu=read_text[1]
		self.bi=read_text[2]
		self.p=read_text[3]
		self.q=read_text[4]
		print("success : loading parameters")


#class SVD++():
	


#class knn():
	


#class integrated():
	



