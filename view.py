
import numpy as np
from scipy.spatial import ConvexHull
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
args = parser.parse_args()

data = np.load(args.i)
xyz = data['xyz']
xwidth = data['xwidth']
ywidth = data['ywidth']
zwidth = data['zwidth']
Nx, Ny, Nz = xyz.shape

xy_view = np.amax(xyz, 2)[::-1,:]
yz_view = np.amax(xyz, 0)
xz_view = np.amax(xyz, 1)[::-1,:]

plt.subplot(1,3,1)
plt.imshow(xy_view, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(yz_view, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(xz_view, cmap='gray')
plt.show()
