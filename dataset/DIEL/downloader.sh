wget http://www.cs.cmu.edu/~lbing/data/emnlp-15-diel/emnlp-15-diel.tar.gz
tar -xvzf emnlp-15-diel.tar.gz
rm -rf emnlp-15-diel.tar.gz
mv emnlp-15-diel/* .
rmdir emnlp-15-diel
rm readme.txt