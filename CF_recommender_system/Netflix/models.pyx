# SVD
import numpy as np
import os
#from numba import vectorize, cuda
cimport numpy as np  # noqa
import data_loader as dl

class SVD():
	def __init__(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1,rating_min=1, rating_max=5 ): 
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
		self.user_index=0
		self.item_index=0

	def reset_parameters(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1,rating_min=1, rating_max=5 ):
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
		self.user_index=0
		self.item_index=0



	def fit(self, trainset,load_parameters=False):
		self.reset_parameters()
		if(not load_parameters):
			self.trainset=trainset
			(mu2,num_user,num_item,user_index,item_index)=dl.get_dataset_info(trainset)
			self.user_index=user_index
			self.item_index=item_index
			self.mu=mu2
			self.bu=np.zeros(int(num_user), np.float64)
			self.bi=np.zeros(int(num_item), np.float64)
			self.p=np.random.normal(self.init_mean, self.init_std, (int(num_user),self.num_factors))
			self.q=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
		if(load_parameters):
			load_parameters()
			(mu2,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)

		trainset=self.trainset
		cdef int num_epochs=self.num_epochs
		cdef np.ndarray[np.double_t] bu=self.bu
		cdef np.ndarray[np.double_t] bi=self.bi
		cdef double mu=self.mu
		cdef int num_factors=self.num_factors
		cdef np.ndarray[np.double_t, ndim=2] p=self.p
		cdef np.ndarray[np.double_t, ndim=2] q=self.q
		cdef double lr=self.lr
		cdef double reg_constant=self.reg_constant

		cdef int user=0
		cdef int item=0
		cdef int rating=0
		cdef double err
		cdef np.ndarray[np.double_t] before_p
		cdef np.ndarray[np.double_t] before_q

		for epoch in range(num_epochs):
			print("epoch: ",epoch)###
			#loop_index=0###
			for user, item, rating in trainset:
				#if(loop_index%100==0):
				#	print("loop: ", loop_index)###
				user=user_index.index(user)
				item=item_index.index(item)
				err=rating-mu-bu[user]-bi[item]-np.dot(q[item],p[user])
				bu[user]+=lr*(err-reg_constant*bu[user])
				bi[item]+=lr*(err-reg_constant*bi[item])
				before_p=p[user]
				before_q=q[item]
				p[user]+=lr*(err*before_q-reg_constant*before_p)
				q[item]+=lr*(err*before_p-reg_constant*before_q)
				#loop_index+=1###


		self.num_epochs=num_epochs
		self.bu=bu
		self.bi=bi
		self.mu=mu
		self.num_factors=num_factors
		self.p=p
		self.q=q
		self.lr=lr
		self.reg_constant=reg_constant


		return 0.0



	def predict(self,dataset):
		result=[]
		cdef int user=0
		cdef int item=0
		cdef int rating=0
		for user, item , rating in dataset:
			prediction=self.mu
			if(user in self.user_index):
				index_user=self.user_index.index(user)
				prediction+=self.bu[index_user]
			if(item in self.item_index):
				index_item=self.item_index.index(item)
				prediction+=self.bi[index_item]
			if((item in self.item_index)and(user in self.user_index)):
				prediction+=np.dot(self.q[index_item],self.p[index_user])
			if(prediction>self.rating_max):
				prediction=self.rating_max
			elif(prediction<self.rating_min):
				prediction=self.rating_min
			else:
				prediction=prediction
			result.append(np.asarray([user,item, rating, prediction], dtype=np.float64))
	#		print(result[-1][2], result[-1][3])
		return np.asarray(result,dtype=object)
		

	def save_parameters(self):
		parameters=[]
		parameters.append(str(np.float64(self.mu)))
		parameters.append('\t'.join([str(b) for b in self.bu]))
		parameters.append('\t'.join([str(b) for b in self.bi]))
		(a,b)=np.shape(self.p)
		parameters.append(str(a)+'\t'+str(b))
		parameters.append('\t'.join([str(aa) for bb in self.p for aa in bb ]))
		(c,d)=np.shape(self.q)
		parameters.append(str(c)+'\t'+str(d))
		parameters.append('\t'.join([str(aa) for bb in self.q for aa in bb ]))
		(e,f)=np.shape(self.trainset)
		parameters.append(str(e)+'\t'+str(f))
		parameters.append('\t'.join([str(aa) for bb in self.trainset for aa in bb ]))
		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVD.txt","w+")
		f.write('\n'.join(parameters))
		f.close()
		print("success : saving parameters")


	def load_parameters(self):

		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVD.txt","r")
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



class SVDpp():
	def __init__(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1,rating_min=1, rating_max=5 ):
		self.trainset=0
		self.mu=0 # global mean mu
		self.bu=0 # user bias
		self.bi=0 # item bias
		self.p=0
		self.q=0
		self.y=0
		self.num_factors=num_factors
		self.init_mean=init_mean
		self.init_std=init_std
		self.lr=lr
		self.reg_constant=reg_constant
		self.num_epochs=num_epochs
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.user_index=0
		self.item_index=0

	def reset_parameters(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1,rating_min=1, rating_max=5 ):
		self.trainset=0
		self.mu=0 
		self.bu=0 
		self.bi=0 
		self.p=0
		self.q=0
		self.y=0
		self.num_factors=num_factors
		self.init_mean=init_mean
		self.init_std=init_std
		self.lr=lr
		self.reg_constant=reg_constant
		self.num_epochs=num_epochs
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.user_index=0
		self.item_index=0



	def fit(self, trainset,load_parameters=False):
		self.reset_parameters()
		if(not load_parameters):
			self.trainset=trainset
			(mu2,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)
			self.user_index=user_index
			self.item_index=item_index
			self.mu=mu2
			self.bu=np.zeros(int(num_user), np.float64)
			self.bi=np.zeros(int(num_item), np.float64)
			self.p=np.random.normal(self.init_mean, self.init_std, (int(num_user),self.num_factors))
			self.q=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
			self.y=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
		if(load_parameters):
			load_parameters()
			(mu2,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)
			self.user_index=user_index
			self.item_index=item_index
			self.mu=mu2

		cdef int num_epochs=self.num_epochs
		cdef np.ndarray[np.double_t] bu=self.bu
		cdef np.ndarray[np.double_t] bi=self.bi
		cdef double mu=self.mu
		cdef int num_factors=self.num_factors
		cdef np.ndarray[np.double_t, ndim=2] p=self.p
		cdef np.ndarray[np.double_t, ndim=2] q=self.q
		cdef double lr=self.lr
		cdef double reg_constant=self.reg_constant
		cdef int user=0
		cdef int item=0
		cdef int rating=0
		cdef np.ndarray[np.double_t, ndim=1] uif=np.zeros(self.num_factors,np.float64)
		cdef np.ndarray[np.double_t, ndim=2] y=self.y
		cdef int old_user=-1
		cdef double err=0
		cdef int prefered_item=0
		cdef int prefered_items=0
		cdef np.ndarray[np.double_t] before_p
		cdef np.ndarray[np.double_t] before_q
		cdef np.ndarray[int, ndim=1] user_prefered_item
		cdef int j
		cdef int k

		for epoch in range(self.num_epochs):

			print("epoch: ",epoch)###
			loop_index=0###
			for user, item, rating in self.trainset:
				if(loop_index%1000==0):
					print("loop: ",loop_index)####
				if(user!=old_user):
					user_prefered_item=np.array([j for (k,j,_) in trainset if k==user ],dtype=np.intc)
				old_user=user
				# convert to index
				user=user_index.index(user)
				item=item_index.index(item)
				uif=np.zeros(num_factors,np.float64)
				for prefered_items in user_prefered_item:
					prefered_item=item_index.index(prefered_items)
					uif+=y[prefered_item]/np.sqrt(len(user_prefered_item))
				err=rating-mu-bu[user]-bi[item]-np.dot(q[item],(p[user]+uif))
				bu[user]+=lr*(err-reg_constant*bu[user])
				bi[item]+=lr*(err-reg_constant*bi[item])
				before_p=p[user]
				before_q=q[item]
				p[user]+=lr*(err*before_q-reg_constant*before_p)
				q[item]+=lr*(err*before_p-reg_constant*before_q)
				for prefered_items in user_prefered_item:
					prefered_item=item_index.index(prefered_items)
					y[prefered_item]+=lr*(err*before_q/np.sqrt(len(user_prefered_item))-reg_constant*y[prefered_item])

				loop_index+=1#####
		print("fitting end")###

		self.num_epochs=num_epochs
		self.bu=bu
		self.bi=bi
		self.mu=mu
		self.num_factors=num_factors
		self.p=p
		self.q=q
		self.lr=lr
		self.reg_constant=reg_constant
		self.y=y


		return 0.0



	def predict(self,dataset):
		print("prediction")###
		result=[]
		cdef np.double_t[:,:] p=self.p
		cdef np.double_t[:,:] q=self.q
		cdef np.double_t[:,:] y=self.y
		cdef int[:] user_prefered_item
		cdef np.double_t[:] bu=self.bu
		cdef np.double_t[:] bi=self.bi
		cdef double mu=self.mu
		cdef int user=0
		cdef int old_user=-1
		cdef int item=0
		cdef int rating=0
		cdef double prediction=0
		for user, item , rating in dataset:
			
			prediction=mu
			if(user in self.user_index):
				index_user=self.user_index.index(user)
				prediction+=bu[index_user]
			if(item in self.item_index):
				index_item=self.item_index.index(item)
				prediction+=bi[index_item]
			if((item in self.item_index)and(user in self.user_index)):
				if(user!=old_user):
					user_prefered_item=np.array([j for (k,j,_) in self.trainset if k==user ],dtype=np.intc)
				old_user=user
				prediction+=np.dot(q[index_item],p[index_user]+(sum(y[self.item_index.index(j)] for j in user_prefered_item) / np.sqrt(len(user_prefered_item))))
			if(prediction>self.rating_max):
				prediction=self.rating_max
			elif(prediction<self.rating_min):
				prediction=self.rating_min
			else:
				prediction=prediction
			result.append(np.asarray([user,item, rating, prediction], dtype=np.float64))
	#		print(result[-1][2], result[-1][3])
		return np.asarray(result,dtype=object)
		

	def save_parameters(self):
		parameters=[]
		parameters.append(str(np.float64(self.mu)))
		parameters.append('\t'.join([str(b) for b in self.bu]))
		parameters.append('\t'.join([str(b) for b in self.bi]))
		(a,b)=np.shape(self.p)
		parameters.append(str(a)+'\t'+str(b))
		parameters.append('\t'.join([str(aa) for bb in self.p for aa in bb ]))
		(c,d)=np.shape(self.q)
		parameters.append(str(c)+'\t'+str(d))
		parameters.append('\t'.join([str(aa) for bb in self.q for aa in bb ]))
		(e,f)=np.shape(self.trainset)
		parameters.append(str(e)+'\t'+str(f))
		parameters.append('\t'.join([str(aa) for bb in self.trainset for aa in bb ]))
		parameters.append(str(np.float64(self.y)))
		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVDpp.txt","w+")
		f.write('\n'.join(parameters))
		f.close()
		print("success : saving parameters")


	def load_parameters(self):

		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVDpp.txt","r")
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
		self.y=float(read_text[9])
		print("success : loading parameters")


class SVDpp_Integrated():
	def __init__(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr1=0.007, lr2=0.007, lr3=0.001, rc6=0.005, rc7=0.015, rc8=0.015, k=300, rating_min=1, rating_max=5 ):
		self.trainset=0
		self.mu=0 # global mean mu
		self.bu=0 # user bias
		self.bi=0 # item bias
		self.p=0
		self.q=0
		self.y=0
		self.num_factors=num_factors
		self.init_mean=init_mean
		self.init_std=init_std
		self.lr1=lr1
		self.lr2=lr2
		self.lr3=lr3
		self.rc6=rc6
		self.rc7=rc7
		self.rc8=rc8
		self.num_epochs=num_epochs
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.user_index=0
		self.item_index=0
		self.k=k

	def reset_parameters(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr1=0.007, lr2=0.007, lr3=0.001, rc6=0.005, rc7=0.015, rc8=0.015, k=300, rating_min=1, rating_max=5 ):
		self.trainset=0
		self.mu=0 
		self.bu=0 
		self.bi=0 
		self.p=0
		self.q=0
		self.y=0
		self.num_factors=num_factors
		self.init_mean=init_mean
		self.init_std=init_std
		self.lr1=lr1
		self.lr2=lr2
		self.lr3=lr3
		self.rc6=rc6
		self.rc7=rc7
		self.rc8=rc8
		self.num_epochs=num_epochs
		self.rating_min=rating_min
		self.rating_max=rating_max
		self.user_index=0
		self.item_index=0
		self.k=k



	def fit(self, trainset,load_parameters=False):
		self.reset_parameters()
		if(not load_parameters):
			self.trainset=trainset
			(mu2,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)
			self.user_index=user_index
			self.item_index=item_index
			self.mu=mu2
			self.bu=np.zeros(int(num_user), np.float64)
			self.bi=np.zeros(int(num_item), np.float64)
			self.p=np.random.normal(self.init_mean, self.init_std, (int(num_user),self.num_factors))
			self.q=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
			self.y=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
		if(load_parameters):
			load_parameters()
			(mu2,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)


		cdef int num_epochs=self.num_epochs
		cdef np.ndarray[np.double_t] bu=self.bu
		cdef np.ndarray[np.double_t] bi=self.bi
		cdef double mu=self.mu
		cdef int num_factors=self.num_factors
		cdef np.ndarray[np.double_t, ndim=2] p=self.p
		cdef np.ndarray[np.double_t, ndim=2] q=self.q
		cdef double lr=self.lr
		cdef double reg_constant=self.reg_constant
		cdef int user=0
		cdef int item=0
		cdef int rating=0
		cdef np.ndarray[np.double_t, ndim=1] uif=np.zeros(self.num_factors,np.float64)
		cdef np.ndarray[np.double_t, ndim=2] y=self.y
		cdef int old_user=-1
		cdef double err=0
		cdef int factors=0
		cdef int prefered_item=0
		cdef np.ndarray[np.double_t] before_p
		cdef np.ndarray[np.double_t] before_q
		cdef int nf=0

		cdef np.ndarray[int, ndim=1] user_prefered_item

		for epoch in range(self.num_epochs):

			print("epoch: ",epoch)###
			loop_index=0###
			for user, item, rating in self.trainset:
				if(loop_index%100==0):
					print("loop: ",loop_index)####
				if(user!=old_user):
					user_prefered_item=np.array([j for (k,j,_) in trainset if k==user ],dtype=np.intc)
				old_user=user
				# convert to index
				user=user_index.index(user)
				item=item_index.index(item)
				uif=np.zeros(num_factors,np.float64)
				for prefered_items in user_prefered_item:
					prefered_item=item_index.index(prefered_items)
					uif+=y[prefered_item]/np.sqrt(len(user_prefered_item))
				err=rating-mu-bu[user]-bi[item]-np.dot(q[item,nf],(p[user]+uif))
				bu[user]+=lr*(err-reg_constant*bu[user])
				bi[item]+=lr*(err-reg_constant*bi[item])
				before_p=p[user]
				before_q=q[item]
				p[user]+=lr*(err*before_q-reg_constant*before_p)
				q[item]+=lr*(err*before_p-reg_constant*before_q)
				for prefered_items in user_prefered_item:
					prefered_item=item_index.index(prefered_items)
					y[prefered_item]+=lr*(err*before_q/np.sqrt(len(user_prefered_item))-reg_constant*y[prefered_item])

				loop_index+=1#####
		print("fitting end")###

		self.num_epochs=num_epochs
		self.bu=bu
		self.bi=bi
		self.mu=mu
		self.num_factors=num_factors
		self.p=p
		self.q=q
		self.lr=lr
		self.reg_constant=reg_constant
		self.y=y


		return 0.0



	def predict(self,dataset):
		print("prediction")###
		result=[]
		cdef np.double_t[:,:] p=self.p
		cdef np.double_t[:,:] q=self.q
		cdef np.double_t[:,:] y=self.y
		cdef int[:] user_prefered_item
		cdef np.double_t[:] bu=self.bu
		cdef np.double_t[:] bi=self.bi
		cdef double mu=self.mu
		cdef int user=0
		cdef int old_user=-1
		cdef int item=0
		cdef int rating=0
		cdef double prediction=0
		for user, item , rating in dataset:
			
			prediction=mu
			if(user in self.user_index):
				index_user=self.user_index.index(user)
				prediction+=bu[index_user]
			if(item in self.item_index):
				index_item=self.item_index.index(item)
				prediction+=bi[index_item]
			if((item in self.item_index)and(user in self.user_index)):
				if(user!=old_user):
					user_prefered_item=np.array([j for (k,j,_) in self.trainset if k==user ],dtype=np.intc)
				old_user=user
				prediction+=np.dot(q[index_item],p[index_user]+(sum(y[self.item_index.index(j)] for j in user_prefered_item) / np.sqrt(len(user_prefered_item))))
			if(prediction>self.rating_max):
				prediction=self.rating_max
			elif(prediction<self.rating_min):
				prediction=self.rating_min
			else:
				prediction=prediction
			result.append(np.asarray([user,item, rating, prediction], dtype=np.float64))
	#		print(result[-1][2], result[-1][3])
		return np.asarray(result,dtype=object)
		

	def save_parameters(self):
		parameters=[]
		parameters.append(str(np.float64(self.mu)))
		parameters.append('\t'.join([str(b) for b in self.bu]))
		parameters.append('\t'.join([str(b) for b in self.bi]))
		(a,b)=np.shape(self.p)
		parameters.append(str(a)+'\t'+str(b))
		parameters.append('\t'.join([str(aa) for bb in self.p for aa in bb ]))
		(c,d)=np.shape(self.q)
		parameters.append(str(c)+'\t'+str(d))
		parameters.append('\t'.join([str(aa) for bb in self.q for aa in bb ]))
		(e,f)=np.shape(self.trainset)
		parameters.append(str(e)+'\t'+str(f))
		parameters.append('\t'.join([str(aa) for bb in self.trainset for aa in bb ]))
		parameters.append(str(np.float64(self.y)))
		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVD.txt","w+")
		f.write('\n'.join(parameters))
		f.close()
		print("success : saving parameters")


	def load_parameters(self):

		f = open(os.path.dirname(os.path.realpath(__file__))+"/parameters/SVD.txt","r")
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
		self.y=float(read_text[9])
		print("success : loading parameters")



