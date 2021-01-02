import pyximport
pyximport.install()
import data_loader as dl
import os
import models
#import modeling
import numpy as np
import accuracy
'''
rmse_list=[]
svd=models.SVD()
svd.fit(dl.loader_100k_1())
estimate=svd.predict(dl.loader_100k_t1())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_2())
estimate=svd.predict(dl.loader_100k_t2())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_3())
estimate=svd.predict(dl.loader_100k_t3())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_4())
estimate=svd.predict(dl.loader_100k_t4())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_5())
estimate=svd.predict(dl.loader_100k_t5())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)


print("SVD RMSE for Fold 1 : ",rmse_list[0])
print("SVD RMSE for Fold 2 : ",rmse_list[1])
print("SVD RMSE for Fold 3 : ",rmse_list[2])
print("SVD RMSE for Fold 4 : ",rmse_list[3])
print("SVD RMSE for Fold 5 : ",rmse_list[4])

'''

rmse_list=[]
svd=models.SVDpp()
svd.fit(dl.loader_100k_1())
estimate=svd.predict(dl.loader_100k_t1())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_2())
estimate=svd.predict(dl.loader_100k_t2())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_3())
estimate=svd.predict(dl.loader_100k_t3())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_4())
estimate=svd.predict(dl.loader_100k_t4())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_5())
estimate=svd.predict(dl.loader_100k_t5())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)


print("SVD++ RMSE for Fold 1 : ",rmse_list[0])
print("SVD++ RMSE for Fold 2 : ",rmse_list[1])
print("SVD++ RMSE for Fold 3 : ",rmse_list[2])
print("SVD++ RMSE for Fold 4 : ",rmse_list[3])
print("SVD++ RMSE for Fold 5 : ",rmse_list[4])




rmse_list=[]
svd=models.SVDpp()
svd.fit(dl.loader_100k_1())
estimate=svd.predict(dl.loader_100k_t1())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_2())
estimate=svd.predict(dl.loader_100k_t2())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_3())
estimate=svd.predict(dl.loader_100k_t3())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_4())
estimate=svd.predict(dl.loader_100k_t4())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)
svd.fit(dl.loader_100k_5())
estimate=svd.predict(dl.loader_100k_t5())
rmse=accuracy.RMSE(estimate)
rmse_list.append(rmse)


print("SVD++ Integrated RMSE for Fold 1 : ",rmse_list[0])
print("SVD++ Integrated RMSE for Fold 2 : ",rmse_list[1])
print("SVD++ Integrated RMSE for Fold 3 : ",rmse_list[2])
print("SVD++ Integrated RMSE for Fold 4 : ",rmse_list[3])
print("SVD++ Integrated RMSE for Fold 5 : ",rmse_list[4])




