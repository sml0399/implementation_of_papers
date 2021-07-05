wget http://www.cs.umd.edu/~sen/lbc-proj/data/WebKB.tgz
tar -xvzf WebKB.tgz
rm -rf WebKB.tgz
mv WebKB/* .
rmdir WebKB 
rm README