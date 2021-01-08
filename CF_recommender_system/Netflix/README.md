# Netflix
## How to use
- 1. python3 setup.py build_ext --inplace
	- (https://riptutorial.com/ko/cython shows how to compile and execute files - written in korean)
- 2. execute the code to test/predict with the model (ex: python3 test_SVD.py )
- Usage Example: (Reading test file is useful. You must use after activating conda environment)
    ```python
    import pyximport
    pyximport.install()
    import data_loader as dl
    import os
    import models
    import numpy as np
    import accuracy
    
    svd=models.SVDpp(num_epochs=20)    # decide model
    data1=dl.loader_100k_1()           # load dataset
    data1=dl.dataset_to_matrix(data1)  # convert dataset to rating matrix
    test1=dl.loader_100k_t1()          # load testset. Conversion to rating matrix is not needed
    svd.load_parameters("svdpp1.txt")  # You can load pretrained parameters 
    svd.fit(data1)                     # train data
    svd.save_parameters("svdpp1.txt")  # save trained parameters
    estimate=svd.predict(test1)        # predict testset
    rmse=accuracy.RMSE(estimate)       # calculate accuracy of prediction
    print(rmse)
    ```	
## Explanation of the files
- Netflix.ppt : ppt file used for seminar(briefly explains the paper)
- .gitignore, _init_.py : _init_.py to make this folder as python module, .gitignore to manage repository
- accuracy.py : python code calculating accuracy(ex: RMSE)
	- RMSE(dataset) : 
		- get RMSE of dataset. Last two info of each input data must be true_value and predicted_value
- data_loader.py : python code that gets data from MovieLens dataset
	- loader_1M() : 
		- load 1M MovieLens dataset
	- loader_100k() : 
		- load 100k MovieLens dataset
	- split_dataset(dataset, num_fold) : 
		- split given dataset into num_fold*[train_set,test_set]
	- get_dataset_info(dataset) : 
		- get the following data - global_mean, number_of_users, number_of_items, users_list, items_list
	- loader_100k_1() ~ loader_100k_5() : 
		- load one of 5-folded dataset(train_set - 80% of total data)
	- loader_100k_t1() ~ loader_100k_t5() : 
		- load one of 5-folded dataset(test_set - 20% of total data)
	- dataset_to_matrix(dataset) :  
		- convert dataset to matrix. 
- install_requirements.sh : shell script for installing necessary libraries
	- install numpy,gcc and cython to anaconda3
- models.pyx : python code that defined models of papers
	- class SVD() : 
		- basic SVD model at paper "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"
	- class SVDpp() : 
		- basic SVDpp model at paper "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"
	- class SVDpp_Integrated() : 
		- not implemented(will be implemented later)
	- parameters for the class initialization
		- num_factors:  
			- default value 50     decide number of factors to be used for model
		- num_epochs:   
			- default value 20     decide number of epochs
		- init_mean:    
			- default value 0      mean for normal distribution to initialize some parameters
		- init_std:     
			- default value 0.1    standard deviation for normal distribution to initialize some parameters
		- lr:           
			- default value 0.005  learning rate
		- reg_constant: 
			- default value 0.1    contstant at regularizing term
		- rating_min:   
			- default value 1      minimum value of rating
		- rating_max:   
			- default value 5      maximum value of rating
	- each class has following functions
		- reset_parameters(): 
			- reset the parameter values. Default value is same as above(class initialization). You can also specify the value just like initialization of class.
		- fit(dataset, load_parameters=False) : 
			- train the model with given data. 'load_parameters=True' means that you will load pretrained parameters. Use load_parameters() function to load them. Default is not loading pretrained parameters. This will show you the RMSE for each epoch during training process
		- predict(dataset) : 
			- predict the ratings of the given trainset. This function will return list of [user,item,real_rating,predicted_rating]
		- save_parameters(name="SVD.txt") : 
			- save parameters to a file at parameters directory. Default name is SVD.txt of SVDpp.txt
		- load_parameters(name="SVD.txt") : 
			- load parameters to a file at parameters directory. Default name is SVD.txt of SVDpp.txt
- setup.py : 
	- used to compile models.pyx ( python3 setup.py build_ext --inplace )
- test_SVD.py : 
	- sample test python code for SVD model
- test_SVDpp.py : 
	- sample test python code for SVD++ model
## Performance:
- The followings are result of training and testing u1.base/u1.test ~ u5.base/u5.test for each model
	- SVD
	```
	1 : fitting_time:  18.361492156982422  RMSE:  0.9555118598821402
	2 : fitting_time:  18.773226261138916  RMSE:  0.94400813272208
	3 : fitting_time:  18.723963260650635  RMSE:  0.936817176058883
	4 : fitting_time:  18.689794301986694  RMSE:  0.9345478183384598
	5 : fitting_time:  18.958670139312744  RMSE:  0.9350943973502772
	```
	- SVDpp
	```
	1 : fitting_time:  105.3489978313446   RMSE:  0.9512272983085271
	2 : fitting_time:  104.22766757011414  RMSE:  0.9398178451260085
	3 : fitting_time:  105.5626630783081   RMSE:  0.9321350503463812
	4 : fitting_time:  106.9869954586029   RMSE:  0.93011797068326
	5 : fitting_time:  109.85181403160095  RMSE:  0.9327594795660497
	```
## Paper
	- [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](https://dl.acm.org/doi/pdf/10.1145/1401890.1401944)
	- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5bNetflix%5d.pdf)
	- Explanation about paper:
	
