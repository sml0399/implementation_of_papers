# accuracy.py
import numpy as np

# dataset that will be given to accuracy calculating functions must have [ [~~~~~, real_value, predicted_value] * number_of_data ] form


def RMSE(dataset):
	return np.sqrt(np.mean([float((true_value - prediction)**2) for (_, _, true_value, prediction) in dataset]))





