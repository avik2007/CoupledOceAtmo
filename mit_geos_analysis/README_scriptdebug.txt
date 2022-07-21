A nice way to debug scripts interactively is to put them in the devel q:

#PBS -q devel at the header

make sure you make it a true bash script: #!/bin/bash at the head

and do chmod u+x "script name".

Then, you can go to a compute node and type in 

qsub -I -q devel -l select=1:ncpus=6:model=has
