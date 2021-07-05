wget http://www.cs.umd.edu/~sen/lbc-proj/data/cora.tgz
tar -xvzf cora.tgz
rm -rf cora.tgz
mv cora/* .
rmdir cora
rm README