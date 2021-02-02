sudo apt-get install unzip

cd 100k_dataset
wget -c "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
unzip ml-100k.zip
cd ml-100k
mv * ../
cd ..
rmdir ml-100k
rm -rf *.zip
rm -rf README

cd ../1M_dataset
wget -c "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
unzip ml-1m.zip
cd ml-1m
mv * ../
cd ..
rmdir ml-1m
rm -rf *.zip
rm -rf README
