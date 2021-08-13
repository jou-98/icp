import open3d as o3d 
import numpy as np 
from time import time
import glob
from sklearn.cluster import KMeans
from pathlib import Path
import copy
import random as rdm
from numpy.linalg import norm as pnorm
from numpy.random import default_rng


# Vanilla sampling method
def downsample(arr1,arr2,max_points,min_points,method='default'):
    pts1 = copy.deepcopy(arr1)
    pts2 = copy.deepcopy(arr2)
    if method == 'default':
        pts1 = pts1[:min_points,:] if pts1.shape[0] >= pts2.shape[0] else pts1
        pts2 = pts2[:min_points,:] if pts2.shape[0] > pts1.shape[0] else pts2
        return pts1,pts2
    if method == 'np_rand':
        rng = default_rng()
        idx = rng.choice(max_points,size=min_points,replace=False)
    elif method == 'py_rand':
        idx = rdm.sample(range(max_points),min_points)
    elif method == 'grid':
        pass 
    elif method == 'fps':
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1)
        pcd2.points = o3d.utility.Vector3dVector(pts2)
        pts1 = fps(pcd1,num_points=15000)
        pts2 = fps(pcd2,num_points=15000)
        max_points = max(pts1.shape[0],pts2.shape[0])
        min_points = min(pts1.shape[0],pts2.shape[0])
        idx = rdm.sample(range(max_points),min_points)
    if pts1.shape[0] == max_points: pts1 = pts1[idx,:]
    if pts2.shape[0] == max_points: pts2 = pts2[idx,:]
    return pts1, pts2


def kmeans_pc(pts,n=1000):
    kmeans = KMeans(n_clusters=1000).fit(pts)
    return kmeans.cluster_centers_

def fps(pcd,num_points=2000,return_pts=True):
    pts = np.asarray(pcd.points)
    n = int(pts.shape[0]/num_points)
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, n)
    if return_pts: return np.asarray(pcd_new.points) 
    return pcd_new

def grid(pts,grid_size=1e-3):
    pass

def iss(pcd):
    pass
"""
files = glob.glob('bunny/data/*.ply')
time_fps = 0
time_km = 0
time_vg = 0

for file in files:

    orig_pcd = o3d.io.read_point_cloud(file,format='ply')
    orig_pts = np.asarray(orig_pcd.points)
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(orig_pcd,
                                                            salient_radius=0.002,
                                                            non_max_radius=0.002,
                                                            gamma_21=0.5,
                                                            gamma_32=0.5)
    print(f'For {file}:')
    print(f'Original pcd has {orig_pts.shape[0]} points and new pcd has {np.array(keypoints.points).shape[0]}')
    #o3d.io.write_point_cloud(file[:-4]+'_kp.ply', keypoints)
    
    n = int(orig_pts.shape[0]/2000)
    path = str(Path(*Path(file).parts[-1:]))
    print(path)
    beg = time()
    pcd = o3d.geometry.PointCloud.uniform_down_sample(orig_pcd, n)
    pts = np.asarray(pcd.points)
    print(f'Subsampled to {pts.shape[0]} points.')
    t = time() - beg
    #print(f'Time taken to subsample through FPS: {round(t,5)}s.')
    o3d.io.write_point_cloud(path[:-4]+'_fps.ply', pcd)
    time_fps += t
    
    beg = time()
    pts = np.asarray(orig_pcd.points)
    pts = kmeans_pc(pts,n=int(pts.shape[0]/20))
    t = time() - beg
    print(f'Time taken to subsample through k-Means: {round(t,5)}s.')
    time_km += t
    
    beg = time()
    orig_pts = np.asarray(orig_pcd.points)
    print(f'Range of coordinates is {np.min(orig_pts)} to {np.max(orig_pts)}')
    pcd = o3d.geometry.PointCloud.voxel_down_sample(orig_pcd, 4e-3)
    pts = np.asarray(pcd.points)
    t = time() - beg
    print(f'Time taken to subsample through voxel grid: {round(t,5)}s.')
    o3d.io.write_point_cloud(path[:-4]+'_vg.ply', pcd)
    time_vg += t
    print(f'FPS and k-Means subsampled to {int(orig_pts.shape[0]/20)} while voxel grid subsamples to {pts.shape[0]}')
    

#print(f'Time taken to subsample through fps: {round(time_fps/len(files),5)}s.')
#print(f'Time taken to subsample through kmeans: {round(time_km/len(files),3)}s.')
#print(f'Time taken to subsample through voxel grid: {round(time_vg/len(files),4)}s.')
"""