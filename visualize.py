
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
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

Nx = 64
Ny = 64
Nz = 64
offset = 0.5

xc = np.linspace(xcmin, xcmax, Nx)
yc = np.linspace(ycmin, ycmax, Ny)
zc = zmax + offset

maxz = np.sqrt(8+(2+offset)**2)

xcg, ycg = np.meshgrid(xc, yc)
xyg = np.stack([xcg, ycg], -1)
sample_pts = np.reshape(xyg, (-1, 2))
sample_pts = np.concatenate([sample_pts, np.ones((sample_pts.shape[0], 1))*zc], -1)

# visual
src_idx = 0
# src_idx = 64*31+32
# src_idx = 64*64-1
src = np.array([sample_pts[src_idx]])
dist = np.linalg.norm(pts - src, ord=2, axis=1, keepdims=True)
R = np.amax(dist) * 10**2
flip = pts + 2 * (R - dist) * (pts - src) / dist
hull = ConvexHull(np.concatenate([flip, src],0))

colors = np.zeros(pts.shape)
n = len(hull.vertices[:-1])
colors[hull.vertices[:-1]] = np.repeat([[1,0.7,0]], n, 0)

visible = hull.vertices[:-1]
vecn = (pts - src) / dist
cos = np.abs(np.sum(vecn[visible] * normals[visible], -1, keepdims=True)) * np.abs(np.expand_dims(vecn[visible,2], -1))
cos = cos / dist[visible]**2
hist = np.histogram((2 * dist[visible]), bins=Nz, range=(0, maxz), weights=cos)

r = np.repeat([[1,0,0]], n, 0)
b = np.repeat([[1,1,0]], n, 0)
cos_scaled = cos / np.amax(cos)
colors[visible] = cos_scaled * r + (1 - cos_scaled) * b

plt.bar(hist[1][:-1], hist[0])
plt.xlabel('scaled time')
plt.ylabel('scaled intensity')
plt.show()

pcd.colors = o3d.utility.Vector3dVector(colors)

wall = o3d.geometry.PointCloud()
wall.points = o3d.utility.Vector3dVector(sample_pts)
wall_colors = np.zeros(sample_pts.shape)
wall_colors[:,2] = np.ones(sample_pts.shape[0])
wall_colors[src_idx] = np.array([1,0,0])
wall.colors = o3d.utility.Vector3dVector(wall_colors)
# src.colors = o3d.utility.Vector3dVector([[0,0,1]])

sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.015)
sphere_vtx = np.array(sphere.vertices)
sphere_vtx = sphere_vtx + src
sphere.vertices = o3d.utility.Vector3dVector(sphere_vtx)
sphere_color = np.repeat([[1,0,0.7]], sphere_vtx.shape[0], 0)
sphere.vertex_colors = o3d.utility.Vector3dVector(sphere_color)

o3d.visualization.draw_geometries([pcd, wall, sphere])
pcd.paint_uniform_color([1,0.7,0])
# pcd.normalize_normals()
o3d.visualization.draw_geometries([pcd], point_show_normal=False)
