import subprocess
import numpy as np
from tqdm import tqdm
info = []
geos = ['sphere','pipe','bunny']
methods = ['mc','halton','test']
level = 10
blocks = int((2**level)/1024)
for amplifier_id in range(5):
    for geo in geos:
        print(geo)
        for method in methods:
            print(method)
            pbar = tqdm(total=2**level)
            for block_id in range(blocks):
                if geo == 'pipe':
                    cmd = './a.out ' + geo + ' 0.6 {} 2 4 1e-3 '.format(amplifier_id) + method + ' 1024 {}'.format(block_id)
                elif geo == 'bunny':
                    cmd = './a.out ' + geo + ' 10 {} 2 3 1e-2 '.format(amplifier_id) + method + ' 1024 {}'.format(block_id)
                else:
                    cmd = './a.out ' + geo + ' 1 {} 2 4 1e-3 '.format(amplifier_id) + method + ' 1024 {}'.format(block_id)
                    
                # invoke process
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

                # Poll process.stdout to show stdout live
                while True:
                  line = process.stdout.readline()
                  if process.poll() is not None:
                    break
                  if line:
                    line = line.decode('utf-8')
                    if line.startswith('smpl_it:'):
                        samples = line.split()[1]
                        pbar.update(1)
                rc = process.poll()
            pbar.close()
