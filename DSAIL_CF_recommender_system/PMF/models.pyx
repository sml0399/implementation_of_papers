# SVD
import numpy as np
import os
#from numba import vectorize, cuda
cimport numpy as np  # noqa
import data_loader as dl
import accuracy
import time
class PMF():
	def __init__(self,num_factors=50,num_epochs=30,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1,rating_min=1, rating_max=5 ): 
		self.trainset=0
		self.mu=0 # global mean mu
		self.bu=0 # user bias
		self.bi=0 # item bias
		self.p=0
		self.q=0
		self.num_factors=num_factors
		self.init_mean=init_mean
		self.init_std=init_std
		self.lr=lr
		self.reg_constant=reg_constant
		self.num_epochs=num_epochs
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.num_user=0
		self.num_item=0

	def reset_parameters(self,num_factors=50,num_epochs=30,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1,rating_min=1, rating_max=5 ):
		self.trainset=0
		self.mu=0 
		self.bu=0 
		self.bi=0 
		self.p=0
		self.q=0
		self.num_factors=num_factors
		self.init_mean=init_mean
		self.init_std=init_std
		self.lr=lr
		self.reg_constant=reg_constant
		self.num_epochs=num_epochs
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.num_user=0
		self.num_item=0


	def fit(self, trainset,load_parameters=False,calculate_accuracy=False): #trainset must be given as matrix
		if(not load_parameters):
			self.trainset=trainset
			self.mu=trainset[trainset>0].mean()
			(num_user,num_item)=np.shape(self.trainset)
			self.num_user=num_user
			self.num_item=num_item
			self.bu=np.zeros(int(num_user), np.float64)
			self.bi=np.zeros(int(num_item), np.float64)
			self.p=np.random.normal(self.init_mean, self.init_std, (int(num_user),self.num_factors))
			self.q=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
		else:
			(num_user,num_item)=np.shape(self.trainset)
			self.num_user=num_user
			self.num_item=num_item
		cdef np.ndarray[int, ndim=2] trainsets=trainset
		cdef int num_epochs=self.num_epochs
		cdef np.ndarray[np.double_t] bu=self.bu
		cdef np.ndarray[np.double_t] bi=self.bi
		cdef double mu=self.mu
		cdef np.ndarray[np.double_t, ndim=2] p=self.p
		cdef np.ndarray[np.double_t, ndim=2] q=self.q
		cdef double lr=self.lr
		cdef double reg_constant=self.reg_constant

		cdef int user=0
		cdef int item=0
		cdef double err
		cdef np.ndarray[np.double_t] before_p
		cdef np.ndarray[np.double_t] before_q
		pairs=np.argwhere(trainsets>0)
		#pairs=zip(trainsets.nonzero())
		if(calculate_accuracy):
			rmse_set=np.array([[user+1,item+1,trainsets[user][item]] for user,item in pairs])
		for epoch in range(num_epochs):
			#print("epoch: ",epoch)###
			for (user,item) in pairs:
				err=trainsets[user][item]-mu-bu[user]-bi[item]-np.dot(q[item],p[user])
				bu[user]+=lr*(err-reg_constant*bu[user])
				bi[item]+=lr*(err-reg_constant*bi[item])
				before_p=p[user]
				before_q=q[item]
				p[user]+=lr*(err*before_q-reg_constant*before_p)
				q[item]+=lr*(err*before_p-reg_constant*before_q)
				#loop_index+=1###
			if(calculate_accuracy):
				predicted=self.predict(rmse_set)
				print("epoch: ",epoch," RMSE: ",accuracy.RMSE(predicted))



		self.bu=bu
		self.bi=bi
		self.p=p
		self.q=q


		return 0.0



	def predict(self,dataset):
		result=[]
		cdef int user=0
		cdef int item=0
		cdef int rating=0
		cdef int index_user=0
		cdef int index_item=0
		cdef double prediction=0
		for user, item , rating in dataset:
			prediction=self.mu
			if(user<=self.num_user):
				prediction+=self.bu[user-1]
			if(item<=self.num_item):
				prediction+=self.bi[item-1]
			if((user<=self.num_user)and(item<=self.num_item)):
				prediction+=np.dot(self.q[item-1],self.p[user-1])
			prediction=min(self.rating_max,prediction)
			prediction=max(self.rating_min,prediction)
			result.append(np.asarray([user,item, rating, prediction], dtype=np.float64))
	#		print(result[-1][2], result[-1][3])
		return np.asarray(result,dtype=object)
		

	def save_parameters(self, name="PMF.txt"):
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


	def load_parameters(self, name="PMF.txt"):

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

