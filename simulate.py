
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
parser.add_argument('-nx', type=int, default=64)
parser.add_argument('-ny', type=int, default=64)
parser.add_argument('-nz', type=int, default=64)
parser.add_argument('-name', type=str, default=None)
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.i)

pts = np.array(pcd.points)
xmin = np.amin(pts[:, 0])
xmax = np.amax(pts[:, 0])
ymin = np.amin(pts[:, 1])
ymax = np.amax(pts[:, 1])
zmin = np.amin(pts[:, 2])
zmax = np.amax(pts[:, 2])
mu = np.mean(pts, 0)
pts = pts - mu
pts = pts / np.max([(xmax-xmin), (ymax-ymin), (zmax-zmin)])
rotate = np.array([[1,0,0],[0,0,1],[0,-1,0]])
pts = pts @ rotate.T
pcd.points = o3d.utility.Vector3dVector(pts)

pcd = pcd.voxel_down_sample(voxel_size=0.01)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

pts = np.array(pcd.points)
normals = np.array(pcd.normals)

xmin = -1
xmax = 1
ymin = -1
ymax = 1
zmin = np.amin(pts[:, 2])
zmax = np.amax(pts[:, 2])

xcmin = (xmin + xmax) / 2 + (xmin - xmax) / 2 * 1
xcmax = (xmin + xmax) / 2 + (xmax - xmin) / 2 * 1
ycmin = (ymin + ymax) / 2 + (ymin - ymax) / 2 * 1
ycmax = (ymin + ymax) / 2 + (ymax - ymin) / 2 * 1

Nx = args.nx
Ny = args.ny
Nz = args.nz
offset = 0.5

xc = np.linspace(xcmin, xcmax, Nx)
yc = np.linspace(ycmin, ycmax, Ny)
zc = zmax + offset

maxz = np.sqrt(8+(2+offset)**2)

xyt = np.zeros((Nx, Ny, Nz))

# # visual
# loc = np.array([[xc[0], yc[63], zc]])
# dist = np.linalg.norm(pts - loc, ord=2, axis=1, keepdims=True)
# R = np.amax(dist) * 10**2
# flip = pts + 2 * (R - dist) * (pts - loc) / dist
# hull = ConvexHull(np.concatenate([flip, loc],0))
#
# colors = np.zeros(pts.shape)
# n = len(hull.vertices[:-1])
# colors[hull.vertices[:-1]] = np.repeat([[1,0,0]], n, 0)
#
# pcd.colors = o3d.utility.Vector3dVector(colors)
#
# src = o3d.geometry.PointCloud()
# src.points = o3d.utility.Vector3dVector(loc)
# src.colors = o3d.utility.Vector3dVector([[0,0,1]])
#
# o3d.visualization.draw_geometries([pcd, src])
# exit()


for i in range(Nx):
    for j in range(Ny):
        loc = np.array([[xc[i], yc[j], zc]])
        vec = pts - loc
        dist = np.linalg.norm(vec, ord=2, axis=1, keepdims=True)
        vecn = vec / dist
        R = np.amax(dist) * 10**2
        flip = pts + 2 * (R - dist) * vecn
        hull = ConvexHull(np.concatenate([flip, loc],0))
        visible = hull.vertices[:-1]
        cos = np.abs(np.sum(vecn[visible] * normals[visible], -1, keepdims=True)) * np.abs(np.expand_dims(vecn[visible,2], -1))
        cos = cos / dist[visible]**2
        xyt[i,j] = np.histogram((2 * dist[visible]), bins=Nz, range=(0, maxz), weights=cos)[0]

xyt = np.sqrt(xyt) * np.array([[np.arange(Nz)]])
filename = 'meas_{:d}x{:d}x{:d}'.format(Nx,Ny,Nz)
if args.name != None:
    filename = args.name + '_' + filename
np.savez(filename, xyt=xyt, xwidth=(xmax-xmin), ywidth=(ymax-ymin), zwidth=maxz)
print('Saved in {:s}.npz'.format(filename))
