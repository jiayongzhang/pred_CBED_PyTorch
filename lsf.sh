#!/bin/bash

#BSUB -P CHP107
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags smt1
#BSUB -N
#BSUB -u zhang.jiayong@foxmail.com
#BSUB -J test
#BSUB -o oe.%J

echo -e "==========================================================" 
echo -e "ENVIRONMENT VARIABLES" 
echo -e "==========================================================" 
printenv
echo -e "==========================================================" 
echo -e "END OF ENVIRONMENT VARIABLES"
echo -e "==========================================================\n" 

echo -e "==========================================================" 
echo -e "JOB STDOUT BEGINS"
echo -e "==========================================================\n" 
date

export OMP_NUM_THREADS=42
#export OMP_DYNAMIC=true
#export OMP_WAIT_POLICY=passive
#export LSB_CHKPNT_DIR=/gpfs/alpine/scratch/z8j/chm147/chkpt
export PATH="/ccs/home/z8j/summit/pytorch-1.0-p3/anaconda3/bin:$PATH"
source ~/summit/pytorch-1.0-p3/source_to_run_pytorch1.0-p3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/summit/pytorch-1.0-p3/9.2.148/lib64/

cd $LS_SUBCWD

#jsrun -n96 /gpfs/alpine/chm147/proj-shared/qe-gpu/bin/pw.x -i qe.in > qe.out
jsrun -n1 -a1 -g6 -c42 --bind none --latency_priority cpu-memory --smpiargs "-gpu" python mpirun.py
#jsrun -n96 /ccs/home/z8j/home_summit/software/qe-gpu/bin/pw.x -i qe.in > qe.out

date
