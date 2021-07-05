# Implementation_of_Papers
## conda environment setup
- executing "conda_create_env.sh"
	```console
	foo@bar:~/Desktop/implementation_of_papers$ ./conda_create_env.sh
	set the name of conda environment: <type_your_new_environment_name_here>
	```
- "conda_create_env.sh" will automatically create anaconda environment with some basic packages. You will be asked to decide the name of the environment. This code is based on anaconda3. The environment contains the followings. You can manually change part of the script depending on your choice.
	- python 3.7
	- cuda 11.1
	- numpy
	- pandas
	- matplotlib
	- sickit-learn
	- seaborn
	- scipy
	- spacy
	- pytorch
	- torchvision 
	- torchaudio
	- torch_geometric
	- networkx
	- graphbrain
	- bzip, gzip, unzip (needed for unzipping downloaded datasets)
- The codes in this repository may not work if you do not use conda environment.

## about directories...
- The following directories are for Internship at [KAIST ISysE DSAIL](https://dsail.kaist.ac.kr/) during 2020.12.28~2021.02.19
	- DSAIL_CF_recommender_system
	- DSAIL_GNN_based
	- DSAIL_random_walk_based
- 'temp' directory is the temporary directory storing implementation of papers that are not classified yet. The implementations here will be moved to proper directory after being classified

