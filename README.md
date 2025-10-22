# Parallel HPE-HGT: Pthreads vs OpenMP vs MPI on DBLP 

This repository contains a reproducible comparative study of **Pthreads, OpenMP, and MPI** for accelerating the **HPE-HGT** model (heterogeneous graph transformer with hybrid positional encodings), targeting **CPU** execution. The primary metrics are:

- **Average time per epoch**
- **Total training time**
- **Peak CPU memory**
- **Accuracy parity** (macro/micro-F1 unchanged vs. serial)

```bash


# 1) (One-time) build the OpenMP attention extension
cd HPEHGT/cpp
python setup_attn_omp.py build_ext --inplace

# 2) Run DBLP with OpenMP via Slurm (edit cpus-per-task to sweep threads)



