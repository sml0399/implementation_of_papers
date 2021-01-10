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

occf=models.OCCF(num_factors=40,num_epochs=50)

train_a=dl.loader_100k_a()
test_a=dl.loader_100k_ta()
train_rating_a=dl.dataset_to_rating(train_a)
test_rating_a=dl.dataset_to_rating(test_a)
occf.fit(train_rating_a,test_rating_a,verbose=True)
occf.save_parameters("occfa.txt")

train_1=dl.loader_100k_1()
test_1=dl.loader_100k_t1()
train_rating_1=dl.dataset_to_rating(train_1)
test_rating_1=dl.dataset_to_rating(test_1)
occf.fit(train_rating_1,test_rating_1)
occf.save_parameters("occf1.txt")

train_2=dl.loader_100k_2()
test_2=dl.loader_100k_t2()
train_rating_2=dl.dataset_to_rating(train_2)
test_rating_2=dl.dataset_to_rating(test_2)
occf.fit(train_rating_2,test_rating_2)
occf.save_parameters("occf2.txt")

train_3=dl.loader_100k_3()
test_3=dl.loader_100k_t3()
train_rating_3=dl.dataset_to_rating(train_3)
test_rating_3=dl.dataset_to_rating(test_3)
occf.fit(train_rating_3,test_rating_3)
occf.save_parameters("occf3.txt")

train_4=dl.loader_100k_4()
test_4=dl.loader_100k_t4()
train_rating_4=dl.dataset_to_rating(train_4)
test_rating_4=dl.dataset_to_rating(test_4)
occf.fit(train_rating_4,test_rating_4)
occf.save_parameters("occf4.txt")

train_5=dl.loader_100k_5()
test_5=dl.loader_100k_t5()
train_rating_5=dl.dataset_to_rating(train_5)
test_rating_5=dl.dataset_to_rating(test_5)
occf.fit(train_rating_5,test_rating_5)
occf.save_parameters("occf5.txt")
