# Netflix
- How to use
	- 1. python3 setup.py build_ext --inplace
		- (https://riptutorial.com/ko/cython shows how to compile and execute files - written in korean)
	- 2. execute the code to test/predict with the model (ex: python3 test.py )

- Explanation of the files
	- accuracy.py : python code calculating accuracy(ex: RMSE)
		- RMSE(dataset) : get RMSE of dataset. Last two info of each input data must be true_value and predicted_value
	- data_loader.py : python code that gets data from MovieLens dataset
		- loader_1M() : load 1M MovieLens dataset
		- loader_100k() : load 100k MovieLens dataset
		- split_dataset(dataset, num_fold) : split given dataset into num_fold*[train_set,test_set]
		- get_dataset_info(dataset) : get the following data - global_mean, number_of_users, number_of_items, users_list, items_list
		- loader_100k_1() ~ loader_100k_5() : load one of 5-folded dataset(train_set - 80% of total data)
		- loader_100k_t1() ~ loader_100k_t5() : load one of 5-folded dataset(test_set - 20% of total data)
	- install_requirements.sh : shell script for installing necessary libraries
		- install numpy and cython to anaconda3
	- models.pyx : python code that defined models of papers
		- class SVD() : 
		- class SVDpp() : 
		- class SVDpp_Integrated() : 
	- setup.py : used to compile models.pyx ( python3 setup.py build_ext --inplace )
		- 
	- test.py : sample test python code
		- 

- Paper
	- Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model
		- https://dl.acm.org/doi/pdf/10.1145/1401890.1401944
	- Matrix Factorization Techniques for Recommender Systems
		- https://datajobs.com/data-science-repo/Recommender-Systems-%5bNetflix%5d.pdf

	- Explanation about paper:
	
