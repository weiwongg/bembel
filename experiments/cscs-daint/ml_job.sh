#!/bin/bash -l
#
#SBATCH -J scatter
#SBATCH --nodes=32
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --constraint=mc
#SBATCH --time=00:40:00
#SBATCH --account=u0



cd /users/whuang/repos/bembel_rough_random_field/experiments
mkdir -p ./data/sphere/0_10_7/mlmc/

#CC -std=c++11 -O3 -I/users/whuang/repos/fmca/ -I/users/whuang/repos/eigen3/ -I../ forward_compute_ML_hybrid_par.cpp \
#-fopenmp \
#-o ml_sample.out

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

srun --nodes=32 --cpu-bind=none ./ml_sample.out -bemlvl 1 -bemdeg 0 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -outputdir ./data/sphere/0_10_7/mlmc/
srun --nodes=32 --cpu-bind=none ./ml_sample.out -bemlvl 1 -bemdeg 1 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -outputdir ./data/sphere/0_10_7/mlmc/
srun --nodes=32 --cpu-bind=none ./ml_sample.out -bemlvl 1 -bemdeg 2 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -outputdir ./data/sphere/0_10_7/mlmc/
srun --nodes=32 --cpu-bind=none ./ml_sample.out -bemlvl 1 -bemdeg 3 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -outputdir ./data/sphere/0_10_7/mlmc/
srun --nodes=32 --cpu-bind=none ./ml_sample.out -bemlvl 1 -bemdeg 4 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -outputdir ./data/sphere/0_10_7/mlmc/
srun --nodes=32 --cpu-bind=none ./ml_sample.out -bemlvl 1 -bemdeg 5 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -outputdir ./data/sphere/0_10_7/mlmc/
srun --nodes=32 --cpu-bind=none ./ml_sample.out -bemlvl 1 -bemdeg 6 -interlvl 0 -interdeg 10 -wavenum 7 -sample mc -setting sphere -outputdir ./data/sphere/0_10_7/mlmc/
