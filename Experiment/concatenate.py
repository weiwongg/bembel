import subprocess
import numpy as np
import pandas as pd
geos = ['sphere']
methods = ['halton']
blocks = int((2**13)/1024)
for geo in geos:
    print(geo)
    for method in methods:
        print(method)
        combined_csv = pd.concat( [ pd.read_csv(geo + '/' + method + '/sol0_block{}.csv'.format(block_id)) for block_id in range(blocks)])
        combined_csv.to_csv(geo + '/' + method + '/' 'sol0.csv', index=False)
        combined_csv = pd.concat( [ pd.read_csv(geo + '/' + method + '/sol1.block{}.csv'.format(block_id)) for block_id in range(blocks)])
        combined_csv.to_csv(geo + '/' + method + '/' 'sol1.csv', index=False)
