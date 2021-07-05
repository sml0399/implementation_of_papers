cd epinion
wget -c "http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2"
bzip2 -d ratings_data.txt.bz2
wget -c "http://www.trustlet.org/datasets/downloaded_epinions/trust_data.txt.bz2"
bzip2 -d trust_data.txt.bz2
rm *.bz2

cd ../epinion_extended
wget -c "http://www.trustlet.org/datasets/extended_epinions/user_rating.txt.gz"
gunzip user_rating.txt.gz
wget -c "http://www.trustlet.org/datasets/extended_epinions/mc.txt.gz"
gunzip mc.txt.gz
wget -c "http://www.trustlet.org/datasets/extended_epinions/rating.txt"
rm *.gz
