 SVD
import numpy as np
import os
from numba import vectorize, cuda
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


	#@vectorize()
	def fit(self, trainset,load_parameters=False):
		self.reset_parameters()
		if(not load_parameters):
			self.trainset=trainset
			(mu,num_user,num_item,user_index,item_index)=dl.get_dataset_info(trainset)
			self.user_index=user_index
			self.item_index=item_index
			self.mu=mu
			self.bu=np.zeros(int(num_user), np.float64)
			self.bi=np.zeros(int(num_item), np.float64)
			self.p=np.random.normal(self.init_mean, self.init_std, (int(num_user),self.num_factors))
			self.q=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
		if(load_parameters):
			load_parameters()
			(mu,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)
		for epoch in range(self.num_epochs):
	#		print("epoch: ",epoch)
			for user, item, rating in self.trainset:
				user=int(user_index.index(user))
				item=int(item_index.index(item))
				err=rating-self.mu-self.bu[user]-self.bi[item]
				for nf in range(self.num_factors):
					err-=self.q[item,nf]*self.p[user,nf]
				self.bu[user]+=self.lr*(err-self.reg_constant*self.bu[user])
				self.bi[item]+=self.lr*(err-self.reg_constant*self.bi[item])
				for nf in range(self.num_factors):
					before_p=self.p[user,nf]
					before_q=self.q[item,nf]
					self.p[user,nf]+=self.lr*(err*before_q-self.reg_constant*before_p)
					self.q[item,nf]+=self.lr*(err*before_p-self.reg_constant*before_q)
		return 0.0


	#@vectorize(['float64(float64)'], target ="cuda")
	def predict(self,dataset):
		result=[]
		for user, item , rating in dataset:
			prediction=self.mu
			if(user in self.user_index):
				index_user=int(self.user_index.index(user))
				prediction+=self.bu[index_user]
			if(item in self.item_index):
				index_item=int(self.item_index.index(item))
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
	def __init__(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr=0.005, reg_constant=0.1,rating_min=1, rating_max=5 ): # default epoch 20
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
		self.user_prefered_item=0

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
		self.user_prefered_item=0


	#@vectorize()
	def fit(self, trainset,load_parameters=False):
		self.reset_parameters()
		user_prefered_item=[]
		if(not load_parameters):
			self.trainset=trainset
			(mu,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)
			self.user_index=user_index
			self.item_index=item_index
			self.mu=mu
			self.bu=np.zeros(int(num_user), np.float64)
			self.bi=np.zeros(int(num_item), np.float64)
			self.p=np.random.normal(self.init_mean, self.init_std, (int(num_user),self.num_factors))
			self.q=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
			self.y=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
		if(load_parameters):
			load_parameters()
			(mu,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)


		for i in range(len(user_index)):
			user_prefered_item.append([j for (k,j,_) in self.trainset if k==user_index[i] ])
		self.user_prefered_item=user_prefered_item

		for epoch in range(self.num_epochs):
			print("epoch: ",epoch)
			for user, item, rating in self.trainset:
				# convert to index
				user=int(user_index.index(user))
				item=int(item_index.index(item))
				uif=np.zeros(self.num_factors,np.float64)
				for prefered_items in self.user_prefered_item[user]:
					prefered_item=int(item_index.index(prefered_items))
					for factors in range(self.num_factors):
						uif[factors]+=self.y[prefered_item, factors]/np.sqrt(len(self.user_prefered_item[user]))
				err=rating-self.mu-self.bu[user]-self.bi[item]
				for nf in range(self.num_factors):
					err-=self.q[item,nf]*(self.p[user,nf]+uif[nf])
				self.bu[user]+=self.lr*(err-self.reg_constant*self.bu[user])
				self.bi[item]+=self.lr*(err-self.reg_constant*self.bi[item])
				for nf in range(self.num_factors):
					before_p=self.p[user,nf]
					before_q=self.q[item,nf]
					self.p[user,nf]+=self.lr*(err*before_q-self.reg_constant*before_p)
					self.q[item,nf]+=self.lr*(err*before_p-self.reg_constant*before_q)
					for prefered_items in self.user_prefered_item[user]:
						prefered_item=int(item_index.index(prefered_items))
						self.y[prefered_item, nf]+=self.lr*(err*before_q/np.sqrt(len(self.user_prefered_item[user]))-reg_constant*self.y[prefered_item, nf])
		return 0.0


	#@vectorize(['float64(float64)'], target ="cuda")
	def predict(self,dataset):
		result=[]
		for user, item , rating in dataset:
			prediction=self.mu
			if(user in self.user_index):
				index_user=int(self.user_index.index(user))
				prediction+=self.bu[index_user]
			if(item in self.item_index):
				index_item=int(self.item_index.index(item))
				prediction+=self.bi[index_item]
			if((item in self.item_index)and(user in self.user_index)):
				prediction+=np.dot(self.q[index_item],self.p[index_user]+(sum(self.y[self.item_index.index(j)] for j in self.user_prefered_item[index_user]) / np.sqrt(len(self.user_prefered_item[index_user]))))
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
	


'''
class SVDpp_Integrated():
	def __init__(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr1=0.007,lr2=0.007,lr3=0.001,rc6=0.005,rc7=0.015,rc8=0.015,k=300 ,rating_min=1, rating_max=5 ):
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
		self.user_prefered_item=0

	def reset_parameters(self,num_factors=100,num_epochs=30,init_mean=0,init_std=0.1, lr1=0.007,lr2=0.007,lr3=0.001,rc6=0.005,rc7=0.015,rc8=0.015,k=300 ,rating_min=1, rating_max=5 ):
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
		self.user_prefered_item=0



	def fit(self, trainset,load_parameters=False):
		self.reset_parameters()
		user_prefered_item=[]
		if(not load_parameters):
			self.trainset=trainset
			(mu,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)
			self.user_index=user_index
			self.item_index=item_index
			self.mu=mu
			self.bu=np.zeros(int(num_user), np.float64)
			self.bi=np.zeros(int(num_item), np.float64)
			self.p=np.random.normal(self.init_mean, self.init_std, (int(num_user),self.num_factors))
			self.q=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
			self.y=np.random.normal(self.init_mean, self.init_std, (int(num_item),self.num_factors))
		if(load_parameters):
			load_parameters()
			(mu,num_user,num_item,user_index,item_index)=dl.get_dataset_info(self.trainset)


		for i in range(len(user_index)):
			user_prefered_item.append([j for (k,j,_) in self.trainset if k==user_index[i] ])
		self.user_prefered_item=user_prefered_item

		for epoch in range(self.num_epochs):
	#		print("epoch: ",epoch)
			for user, item, rating in self.trainset:
				# convert to index
				user=int(user_index.index(user))
				item=int(item_index.index(item))
				uif=np.zeros(self.num_factors,np.float64)
				for prefered_items in self.user_prefered_item[user]:
					prefered_item=int(item_index.index(prefered_items))
					for factors in range(self.num_factors):
						uif[factors]+=self.y[prefered_item, factors]/np.sqrt(len(self.user_prefered_item[user]))
				err=rating-self.mu-self.bu[user]-self.bi[item]
				for nf in range(self.num_factors):
					err-=self.q[item,nf]*(self.p[user,nf]+uif[nf])
				self.bu[user]+=self.lr*(err-self.reg_constant*self.bu[user])
				self.bi[item]+=self.lr*(err-self.reg_constant*self.bi[item])
				for nf in range(self.num_factors):
					before_p=self.p[user,nf]
					before_q=self.q[item,nf]
					self.p[user,nf]+=self.lr*(err*before_q-self.reg_constant*before_p)
					self.q[item,nf]+=self.lr*(err*before_p-self.reg_constant*before_q)
					for prefered_items in self.user_prefered_item[user]:
						prefered_item=int(item_index.index(prefered_items))
						self.y[prefered_item, nf]+=self.lr*(err*before_q/np.sqrt(len(self.user_prefered_item[user]))-self.reg_constant*self.y[prefered_item, nf])
		return 0.0



	def predict(self,dataset):
		result=[]
		for user, item , rating in dataset:
			prediction=self.mu
			if(user in self.user_index):
				index_user=int(self.user_index.index(user))
				prediction+=self.bu[index_user]
			if(item in self.item_index):
				index_item=int(self.item_index.index(item))
				prediction+=self.bi[index_item]
			if((item in self.item_index)and(user in self.user_index)):
				prediction+=np.dot(self.q[index_item],self.p[index_user]+(sum(self.y[self.item_index.index(j)] for j in self.user_prefered_item[index_user]) / np.sqrt(len(self.user_prefered_item[index_user]))))
			if(prediction>self.rating_max):
				prediction=self.rating_max
			elif(prediction<self.rating_min):
				prediction=self.rating_min
			else:
				prediction=prediction
			result.append(np.asarray([user,item, rating, prediction], dtype=np.float64))
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
	

'''

