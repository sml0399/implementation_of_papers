read -p 'set the name of conda environment: ' envname
conda create -y -n $envname python=3.7
eval "$(conda shell.bash hook)"
conda activate $envname
conda install -y -c anaconda scikit-learn
conda install -y -c anaconda numpy
conda install -y -c conda-forge matplotlib
conda install -y -c anaconda pandas
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -y -c pytorch
conda install -y pytorch-geometric -c rusty1s -c conda-forge
conda install -y -c anaconda networkx
conda install -y -c anaconda seaborn
conda install -y -c anaconda scipy
