#test.py
import pyximport
pyximport.install()
import data_loader as dl
import os
import models
#import modeling
import numpy as np
import accuracy
import time



svd=models.SVD(num_epochs=20)
fit_time=[]
rmse_record=[]
data1=dl.loader_100k_1()
data1=dl.dataset_to_matrix(data1)
test1=dl.loader_100k_t1()
data2=dl.loader_100k_2()
data2=dl.dataset_to_matrix(data2)
test2=dl.loader_100k_t2()
data3=dl.loader_100k_3()
data3=dl.dataset_to_matrix(data3)
test3=dl.loader_100k_t3()
data4=dl.loader_100k_4()
data4=dl.dataset_to_matrix(data4)
test4=dl.loader_100k_t4()
data5=dl.loader_100k_5()
data5=dl.dataset_to_matrix(data5)
test5=dl.loader_100k_t5()


start_time=time.time()
svd.fit(data1)
end_time=time.time()
fit_time.append(str(end_time-start_time))
svd.save_parameters("svd1.txt")
estimate=svd.predict(test1)
rmse=accuracy.RMSE(estimate)
rmse_record.append(rmse)
start_time=time.time()
svd.fit(data2)
end_time=time.time()
fit_time.append(str(end_time-start_time))
svd.save_parameters("svd2.txt")
estimate=svd.predict(test2)
rmse=accuracy.RMSE(estimate)
rmse_record.append(rmse)
start_time=time.time()
svd.fit(data3)
end_time=time.time()
fit_time.append(str(end_time-start_time))
svd.save_parameters("svd3.txt")
estimate=svd.predict(test3)
rmse=accuracy.RMSE(estimate)
rmse_record.append(rmse)
start_time=time.time()
svd.fit(data4)
end_time=time.time()
fit_time.append(str(end_time-start_time))
svd.save_parameters("svd4.txt")
estimate=svd.predict(test4)
rmse=accuracy.RMSE(estimate)
rmse_record.append(rmse)
start_time=time.time()
svd.fit(data5)
end_time=time.time()
fit_time.append(str(end_time-start_time))
svd.save_parameters("svd5.txt")
estimate=svd.predict(test5)
rmse=accuracy.RMSE(estimate)
rmse_record.append(rmse)

for i in range(5):
	print(i+1,": fitting_time: ", fit_time[i]," RMSE: ", rmse_record[i])

