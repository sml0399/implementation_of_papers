import pyximport
pyximport.install()
import data_loader as dl
import os
import models
#import modeling
import numpy as np
import accuracy
svd=models.SVDpp(num_factors=50,num_epochs=20)
svd.fit(dl.loader_100k_3())
svd.save_parameters("svdpp3.txt")
estimate=svd.predict(dl.loader_100k_t3())
rmse=accuracy.RMSE(estimate)
print("SVD++ RMSE for Fold 3 : ",rmse)
