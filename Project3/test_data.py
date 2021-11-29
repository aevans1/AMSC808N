import numpy as np
import matplotlib.pyplot as plt
import diffusion_map as dmap
import datetime
from mpl_toolkits import mplot3d

inData = np.load("FaceData.npz")
data = inData["data"]
colors = inData["colors"]
print(data.shape)
print(colors.shape)

N = data.shape[0]

# Plotting some images
#idx_list = np.arange(1, N, 200)
#for idx in idx_list:
#    print(data[idx, :])
#    image = np.reshape(data[idx, :], [40, 40], order='F')
#    plt.figure()
#    plt.imshow(image, origin="lower")
#plt.show()

#eps_list = [0.1, 1.0, 10.0, 100.0, 1000.0]
#eps_list = [10.0, 100.0, 1000.0]
#eps_list = np.arange(0.3, 0.5, 0.02)
#eps_list = np.arange(, 1.0, 0.1)
#eps_list = np.arange(1.0, 10.0, 1)
#eps_list = np.arange(1.0, 10.0, 1)
#eps_list = np.arange(1920.0, 1950,10)
eps_list = np.arange(100.0, 2000.0, 400)
for eps in eps_list:
    my_dmap = dmap.DiffusionMap(alpha=1, epsilon=eps, num_evecs=3)
    my_dmap.fit(data.T)
    evecs = my_dmap.evecs
    evals = my_dmap.evals
    dcoords = my_dmap.dmap
    #plt.scatter(evecs[:, 0], evecs[:, 1], s=1.0)
    #plt.scatter(dcoords[:, 0], dcoords[:, 2], s=1.0)
    #plt.figure()
    fig = plt.figure()
    ax = plt.axes(projection='3d') 
    ax.scatter3D(evecs[:, 0], evecs[:, 1], evecs[:, 2], cmap='Greens')
    #plt.scatter([0, 1], evals)
plt.show()