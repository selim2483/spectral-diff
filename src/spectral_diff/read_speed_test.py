import os
import time
import h5py


base_path = os.getcwd().split('/sollivie/')[0]
t1 = time.time()
for idx in range(6):
    path = os.path.join(base_path, 'sollivie', 'datasets', f'cloud_{idx}.hdf5')
    t2 = time.time()
    hf = h5py.File(path, 'r')
    print(f'Opening file time: {time.time() - t2:.4f}')

    t2 = time.time()
    data = hf['data'][0]
    print(f'Reading cube + slicing time: {time.time() - t2:.4f}')

    # t2 = time.time()
    # frame = hf['frame_000']
    # print(f'Reading frame time: {time.time() - t2:.4f}')

    t2 = time.time()
    print(f'Total time : {t2 - t1:.4f}')
    print(f'Mean time : {(t2 - t1) / (idx + 1):.4f}')