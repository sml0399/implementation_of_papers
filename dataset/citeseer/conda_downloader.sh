wget http://www.cs.umd.edu/~sen/lbc-proj/data/citeseer.tgz
tar -xvzf citeseer.tgz
rm -rf citeseer.tgz
mv citeseer/* .
rmdir citeseer
rm README