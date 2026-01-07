import numpy as np
import torch
import time

t1 = time.time()
data = torch.load('/tmp_user/juno/sollivie/datasets/cloudy/cloud_1061.pt')
print(f'time : {time.time() - t1}')