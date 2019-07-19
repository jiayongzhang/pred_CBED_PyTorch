#! /bin/bash

#PBS -A sns
#PBS -q batch
#PBS -m ea
#PBS -M zhang.jiayong@foxmail.com
#PBS -j oe
#PBS -o oe.$PBS_JOBID
#PBS -l qos=long
#PBS -W group_list=cades-virtues
#PBS -l walltime=300:00:00
#PBS -l nodes=1:ppn=32
#PBS -N CBED

#export MODULEPATH=/software/dev_tools/swtree/or-condo/modulefiles:$MODULEPATH

#module load PE-intel
#module load mkl
module load anaconda3/5.1.0-pe3
module list

export OMP_NUM_THREADS=32

cd $PBS_O_WORKDIR

date

#tar xvf 10.13139_OLCF_1510313.tar 
#python run.py
python countSpace.py
wait

date
