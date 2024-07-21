#!/bin/bash

module reset

# Load modules
module load gcc/9.1.0
module load python3/3.9
module load cuda/12

source venv/bin/activate

# test env
# --------
# echo 'which python -> venv'
# which python

# echo 'test_pytorch.py -> random tensor'
# python test/test_pytorch.py

# echo 'test_pytorch_cuda_gpu.py -> True if GPU'
# python test/test_pytorch_cuda_gpu.py
