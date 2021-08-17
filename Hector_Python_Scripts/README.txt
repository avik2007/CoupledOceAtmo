How to install python virtual environments

1) create a folder to place python environment
   Example:
   cd /nobackup/htorresg/
   mkdir .conda/envs
   
2) Load modules related to miniconda
   module use -a /swbuild/analytix/tools/modulefiles
   module load miniconda3/v4
   export CONDA_ENVS_PATH=/nobackup/htorresg/.conda/envs

3) create python env
   option 1:
   conda create --name xmitgcm3.yml --clone base

   option 2:
   conda env create -f xmitgcm3.yml

   

4) Activate the virtual environment
   source activate xmitgcm3


5) Deactivate 
   conda deactibate NAME


How to install python packages
1) Activate python environment
   module use -a /swbuild/analytix/tools/modulefiles
   module load miniconda3/v4
   export CONDA_ENVS_PATH=/nobackup/htorresg/.conda/envs 
   source activate NAME

2) Install packages
   pip install --use name_package





