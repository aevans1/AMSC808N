import sys 

import numpy as np
import scipy.io as sio

# Load the data
#fname = 'FaceData.mat'
fname = 'ScurveData_noise005.mat'
contents = sio.loadmat(fname)
    
# Check what the keys are, each relevant one should be after __version__
print(sorted(contents.keys()))

fname = 'ScurveData_noise005'    
data = contents['data3']
np.savez(fname, data=data)