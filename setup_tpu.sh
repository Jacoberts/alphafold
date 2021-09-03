
# Setup conda environment.
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/anaconda_installer.sh
bash ~/anaconda_installer.sh -f -b -p $HOME/anaconda
export PATH="$HOME/anaconda/bin:$PATH"
source "$HOME/anaconda/bin/activate"

conda create --name alphafold python==3.8
conda update -n base conda
conda activate alphafold

# Conda dependencies.
conda install -y -c conda-forge openmm==7.5.1 cudnn==8.2.1.32 cudatoolkit==11.0.3 pdbfixer==1.7
conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04

# Pip dependencies.
pip install --upgrade pip
pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow==2.5.0
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Patch one of the pip dependencies...
alphafold_path="$HOME/alphafold"
cd ~/anaconda3/envs/alphafold/lib/python3.8/site-packages/ && patch -p0 < $alphafold_path/docker/openmm.patch
cd $HOME/alphafold

# Setup the db disk.
sudo mkdir -p /mnt/disks/afdbs
sudo mount -o discard,defaults /dev/sdb /mnt/disks/afdbs
sudo chmod a+w /mnt/disks/afdbs

# Setup working directories.
mkdir $HOME/out
mkdir $HOME/tmp
mkdir /tmp/jackhmmer
mkdir /tmp/hhsearch
