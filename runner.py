from v3.startup import run, pre_process
import sys
import random
import torch
import numpy as np


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# clear && nvidia-smi && free
# git pull && clear && python3 runner.py | tee log.txt


pre_process()


print('Start Trainig')
sys.stdout.flush()

config= [
    [48, (0.9, 0.999), 1e-3, 100_000, 15],
]

for c in config:
    print("\u001B[36mbatch_size={}, betas={}, lr={}, epochs={}, stop_after={} \x1b[0m".format(c[0], c[1], c[2], c[3], c[4]))
    sys.stdout.flush()

    run(batch_size=c[0], betas=c[1], lr=c[2], epochs=c[3], stop_after=c[4])