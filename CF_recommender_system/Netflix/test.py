import data_loader as dl
data=dl.loader_100k()
print(data)
dataset=dl.split_dataset(data)
print(dataset)
