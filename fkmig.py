
import numpy as np
from scipy.spatial import ConvexHull
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
parser.add_argument('-name', type=str, default=None)
args = parser.parse_args()

data = np.load(args.i)
xyt = data['xyt']
xwidth = data['xwidth']
ywidth = data['ywidth']
zwidth = data['zwidth']
Nx, Ny, Nz = xyt.shape

print('reconstructing...')

xyt_pad = np.pad(xyt, ((0, Nx), (0, Ny), (0, Nz)),
                 'constant', constant_values=0)
Fxyt = fftshift(fftn(xyt_pad))

kx = np.linspace(-1, 1, 2*Nx)
ky = np.linspace(-1, 1, 2*Ny)
kz = np.linspace(-1, 1, 2*Nz)
kxg, kyg, kzg = np.meshgrid(kx, ky, kz)

Fxyt_interp = RegularGridInterpolator((kx, ky, kz), Fxyt, bounds_error=False, fill_value=0)
Fxyz = Fxyt_interp(np.stack([kxg, kyg, np.sqrt(np.abs(((Nx*zwidth/xwidth/Nz/2)**2)*kxg**2+((Ny*zwidth/ywidth/Nz/2)**2)*kyg**2+kzg**2))], -1))
Fxyz = Fxyz * (kzg > 0)
Fxyz = Fxyz * np.abs(kzg) / np.sqrt(np.abs(((Nx*zwidth/xwidth/Nz/2)**2)*kxg**2+((Ny*zwidth/ywidth/Nz/2)**2)*kyg**2+kzg**2))
xyz = ifftn(ifftshift(Fxyz))
xyz = np.abs(xyz)**2
xyz = xyz[:Nx, :Ny, :Nz]

filename = 'rec_{:d}x{:d}x{:d}'.format(Nx,Ny,Nz)
if args.name != None:
    filename = args.name + '_' + filename
np.savez(filename, xyz=xyz, xwidth=xwidth, ywidth=ywidth, zwidth=zwidth)
print('Saved in {:s}.npz'.format(filename))

xy_view = np.amax(xyz, 2)[::-1,:]
yz_view = np.amax(xyz, 0)
xz_view = np.amax(xyz, 1)[::-1,:]

# depth = (Nz - 1 - np.argmax(xyz, -1)[::-1,:]).astype(float) / (Nz - 1)
# intensity = np.amax(xyz, -1)[::-1,:]
# background = intensity < 0.05 * np.amax(intensity)
# depth[background] = np.zeros(np.sum(background))

plt.subplot(1,3,1)
plt.imshow(xy_view, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(yz_view, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(xz_view, cmap='gray')
plt.show()
