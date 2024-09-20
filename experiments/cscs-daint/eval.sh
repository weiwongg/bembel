#!/bin/bash -l
#
#SBATCH -J scatter
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --constraint=mc
#SBATCH --time=00:30:00
#SBATCH --account=u0


cd /users/whuang/repos/bembel_rough_random_field/experiments

#CC -std=c++11 -O3 -I/users/whuang/repos/fmca/ -I/users/whuang/repos/eigen3/ -I../ forward_eval_ML.cpp \
#-fopenmp \
#-o eval.out

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

./eval.out -bemlvl 1 -bemdeg 6 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -inputdir /users/whuang/backup/data/sphere/0_10_7/mlmc/ -refdir /users/whuang/backup/data/sphere/0_10_7/mc/
