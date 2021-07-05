read -p 'set the name of conda environment: ' envname
read -p 'decide cuda version between 10.2 and 11.1: ' cuda_version
conda create -y -n $envname python=3.7
eval "$(conda shell.bash hook)"
conda activate $envname
if [ $cuda_version = "10.2" ]
then
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -y -c pytorch
elif [ $cuda_version = "11.1" ]
then
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -y -c pytorch -c nvidia
else
    echo "ERROR. Cuda will not be installed"
fi
conda install -y -c anaconda scikit-learn
conda install -y -c anaconda numpy
conda install -y -c conda-forge matplotlib
conda install -y -c anaconda pandas
conda install -y pytorch-geometric -c rusty1s -c conda-forge
conda install -y -c anaconda networkx
conda install -y -c anaconda seaborn
conda install -y -c anaconda scipy
conda install -y -c conda-forge spacy
conda install -y -c conda-forge graphbrain
conda install -y bzip2
conda install -y -c ostrokach gzip
conda install -y -c conda-forge unzip
