#!/bin/bash -l
#
#SBATCH -J scatter
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --constraint=mc
#SBATCH --time=01:00:00
#SBATCH --account=u0b

module load cray/21.12
#module swap PrgEnv-cray PrgEnv-intel
#module load EasyBuild-custom
#module load Core/Eigen/3.4.0
#module load Core/METIS/5.1.0

cd /users/whuang/repos/bembel_rough_random_field/experiments

#CC -std=c++11 -O3 -I/users/whuang/repos/fmca/ -I../ forward_eval_ML.cpp \
#-fopenmp -lblas -llapack -lmetis -DEIGEN_USE_MKL_ALL -lm -lgfortran -lmkl_intel_lp64 \
#-lmkl_lapack95_lp64 -lmkl_sequential -lmkl_core \
#-o eval.out

#CC -std=c++11 -O3 -I/users/whuang/repos/fmca/ -I/users/whuang/repos/eigen3/ -I../ forward_eval_ML.cpp \
#-fopenmp \
#-o eval.out

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

./eval.out -bemlvl 1 -bemdeg 5 -interlvl 0 -interdeg 8 -wavenum 5 -sample mc -setting torus -inputdir ./data/torus/0_8_5/mlmc/ -refdir ./data/torus/0_8_5/mlmc_ref/
