# Parallel HPE-HGT: OpenMP vs CUDA vs Serial on DBLP 

This repository contains a reproducible comparative study of ** MPI and CUDA** for accelerating the **HPE-HGT** model (heterogeneous graph transformer with hybrid positional encodings). The primary metrics are:

- **Average time per epoch**
- **Total training time**
- **Peak CPU memory**
- **Accuracy parity** (macro/micro-F1 unchanged vs. serial)

```bash





