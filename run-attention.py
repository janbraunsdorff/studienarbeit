from attention.startup import run, pre_process
import sys
import random
import torch
import numpy as np


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# clear && nvidia-smi && free
# git pull && clear && python3 run-attention.py | tee log.txt


pre_process()


print('Start Trainig')
sys.stdout.flush()

config= [
    [32, 1e-6],
]

for c in config:
    print(c)
    run(batch_size=c[0], lr=c[1])