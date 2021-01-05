import pyximport
pyximport.install()
import data_loader as dl
import os
import models
#import modeling
import numpy as np
import accuracy
svd=models.SVDpp(num_factors=50,num_epochs=20)
svd.fit(dl.loader_100k_5())
svd.save_parameters("svdpp5.txt")
estimate=svd.predict(dl.loader_100k_t5())
rmse=accuracy.RMSE(estimate)
print("SVD++ RMSE for Fold 5 : ",rmse)
