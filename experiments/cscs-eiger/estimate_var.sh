#!/bin/bash -l
#
#SBATCH -J scatter
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --constraint=mc
#SBATCH --time=03:00:00
#SBATCH --account=u0b

module load cray/21.12
#module swap PrgEnv-cray PrgEnv-intel
#module load EasyBuild-custom
#module load Core/Eigen/3.4.0
#module load Core/METIS/5.1.0

cd /users/whuang/repos/bembel_rough_random_field/experiments
#mkdir -p data/brick/0_14_1/bem/
#mkdir plot

#CC -std=c++11 -O3 -I/users/whuang/repos/fmca/ -I../ estimate_variance.cpp \
#-fopenmp -lblas -llapack -lmetis -DEIGEN_USE_MKL_ALL -lm -lgfortran -lmkl_intel_lp64 \
#-lmkl_lapack95_lp64 -lmkl_sequential -lmkl_core

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

./estimate.out -setting david
