# SVD
import numpy as np
import os
#from numba import vectorize, cuda
cimport numpy as np  # noqa
import data_loader as dl
import time
class OCCF():
	def __init__(self,num_factors=40,num_epochs=30,init_mean=0,init_std=0.01, reg_constant=0.002,rating_min=1, rating_max=5 ): 
		self.trainset=0
		self.preference=0
		self.confidence=0
		self.testset=0
		self.num_factors=num_factors
		self.num_epochs=num_epochs
		self.init_mean=init_mean
		self.init_std=init_std
		self.reg_constant=reg_constant
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.x=0
		self.y=0
		

	def reset_parameters(self,num_factors=40,num_epochs=30,init_mean=0,init_std=0.01, reg_constant=0.002,rating_min=1, rating_max=5 ):
		self.trainset=0
		self.preference=0
		self.confidence=0
		self.testset=0
		self.num_factors=num_factors
		self.num_epochs=num_epochs
		self.init_mean=init_mean
		self.init_std=init_std
		self.reg_constant=reg_constant
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.x=0
		self.y=0

	def fit(self, trainset,testset,load_parameters=False,verbose=False): #trainset must be given as matrix
		occf_rank=[]
		self.occf_rank=occf_rank
		if(not load_parameters):
			self.trainset=trainset
			self.testset=testset
			(num_users,num_items)=np.shape(trainset)
			self.num_user=num_users
			self.num_item=num_items
			self.x=np.random.normal(self.init_mean, self.init_std, (int(num_users),self.num_factors))
			self.y=np.random.normal(self.init_mean, self.init_std, (int(num_items),self.num_factors))
		else:
			(num_users,num_items)=np.shape(self.trainset)
			self.num_user=num_users
			self.num_item=num_items
		
		self.preference=dl.rating_to_preference(trainset)
		self.confidence=dl.rating_to_confidence(testset)


		cdef np.ndarray[int, ndim=2] trainsets=self.trainset
		cdef np.ndarray[int, ndim=2] testsets=self.testset
		cdef np.ndarray[int, ndim=2] preferences=self.preference
		cdef np.ndarray[np.double_t, ndim=2] confidences=self.confidence
		cdef int num_epochs=self.num_epochs
		cdef int num_factors=self.num_factors
		cdef int epoch=0;
		cdef np.ndarray[np.double_t, ndim=2] x=self.x
		cdef np.ndarray[np.double_t, ndim=2] y=self.y
		cdef double reg_constant=self.reg_constant
		cdef int user=0
		cdef int item=0
		cdef int num_user=self.num_user
		cdef int num_item=self.num_item
		cdef np.ndarray[np.double_t,ndim=2] c_user=np.zeros((num_item,num_item),dtype=np.float64)
		cdef np.ndarray[np.double_t,ndim=2] c_item=np.zeros((num_user,num_user),dtype=np.float64)
		cdef np.ndarray[np.double_t,ndim=2] predicted=np.zeros((num_user,num_item),dtype=np.float64)
		#cdef np.ndarray[np.double_t,ndim=1] partial_rank
		cdef int sum_testset=testsets.sum()
		occf_rank=self.occf_rank
		total_time=0;
		for epoch in range(num_epochs):
			start_time=time.time()
			for user in range(num_user):
				c_user=np.zeros((num_item,num_item),dtype=np.float64)
				for i in range(num_item):
					c_user[i,i]=confidences[user,i]
				x[user,:]=np.linalg.solve(np.matmul(np.matmul(y.T,c_user),y)+reg_constant*np.eye(num_factors),np.matmul(np.matmul(y.T,c_user),preferences[user,:].T))
			for item in range(num_item):
				c_item=np.zeros((num_user,num_user),dtype=np.float64)
				for u in range(num_user):
					c_item[u,u]=confidences[u,item]
				y[item,:]=np.linalg.solve(np.matmul(np.matmul(x.T,c_item),x)+reg_constant*np.eye(num_factors),np.matmul(np.matmul(x.T,c_item),preferences[:,item]))
			time_took=time.time()-start_time
			total_time=total_time+time_took
			rank_weighted_sum_ratings=0;
			predicted=np.array(np.matmul(x,y.T),dtype=np.float64)
			for i in range(num_user):
				for j in range(num_item):
					if testsets[i,j]!=0:
						sorted_result=sorted([predicted[i,k] for k in range(num_item) if trainsets[i,k]==0])
						rank_weighted_sum_ratings+=testsets[i,j]*(1-(sorted_result.index(predicted[i,j])/len(sorted_result)))

			occf_rank.append(rank_weighted_sum_ratings/sum_testset)
			if(verbose):
				print("fit time for epoch ",epoch,": ",time_took,", rank at current epoch: ",occf_rank[-1] )

		print("total fit time for data: ",total_time,", rank at final epoch: ", occf_rank[-1])		

		self.x=x
		self.y=y


		return occf_rank



	def predict(self):
		cdef np.ndarray[np.double_t, ndim=2] x=self.x
		cdef np.ndarray[np.double_t, ndim=2] y=self.y
		return np.matmul(x,y.T)
		

	def save_parameters(self, name="OCCF.txt"):
		parameters=[]
		(a,b)=np.shape(self.x)
		parameters.append(str(a)+'\t'+str(b))
		parameters.append('\t'.join([str(aa) for bb in self.x for aa in bb ]))			# x
		(c,d)=np.shape(self.y)
		parameters.append(str(c)+'\t'+str(d))
		parameters.append('\t'.join([str(aa) for bb in self.y for aa in bb ]))			# y
		(e,f)=np.shape(self.trainset)
		parameters.append(str(e)+'\t'+str(f))
		parameters.append('\t'.join([str(aa) for bb in self.trainset for aa in bb ]))		# trainset
		(g,h)=np.shape(self.testset)
		parameters.append(str(g)+'\t'+str(h))
		parameters.append('\t'.join([str(aa) for bb in self.testset for aa in bb ]))		# testset
		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/"+name,"w+")
		f.write('\n'.join(parameters))
		f.close()
		print("success : saving parameters")


	def load_parameters(self, name="OCCF.txt"):

		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/"+name,"r")
		read_text=f.read()
		f.close()
		read_text=read_text.split('\n')
		(a,b)=read_text[0].split('\t')
		(c,d)=read_text[2].split('\t')
		(e,f)=read_text[4].split('\t')
		(g,h)=read_text[6].split('\t')
		self.x=np.asarray(read_text[1].split('\t'),dtype=np.float).reshape(int(a),int(b) )
		self.y=np.asarray(read_text[3].split('\t'),dtype=np.float).reshape(int(c),int(d) )
		self.trainset=np.asarray(read_text[5].split('\t'),dtype=np.float).reshape(int(e),int(f) )
		self.testset=np.asarray(read_text[7].split('\t'),dtype=np.float).reshape(int(g),int(h) )
		print("success : loading parameters")



