import numpy as np 
import open3d as o3d 
from utils import * 
from sampling import downsample
from sklearn.neighbors import NearestNeighbors as NN
from scipy.optimize import linear_sum_assignment as LSA
import random as rdm


def preprocess_point_cloud(pcd,voxel_size=0.001):

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def prepare_dataset(src,dst,voxel_size):
    #print(":: Load two point clouds and disturb initial pose.")
    source,pts1 = o3d_read_pc(src)
    target,pts2 = o3d_read_pc(dst)
    n_points = min(pts1.shape[0],pts2.shape[0])
    pts1,pts2 = downsample(pts1,pts2,method='fps',n_fps=0.2)
    source = o3d_instance(pts1)
    target = o3d_instance(pts2)
    print(np.asarray(source.points).shape[0])
    print(np.asarray(target.points).shape[0])

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_fpfh, target_fpfh



voxel_size = 0.001
which = 'bunny'
file1 = 'bun000'
file2 = 'bun045'
suffix=''
f1,f2=file1,file2
if which == 'bunny':
    suffix = 'bunny/data/'
    if 'bun' in file1: f1 = file1[-3:]
    if 'bun' in file2: f2 = file2[-3:]
elif which == 'happy_stand':
    suffix = 'happy_stand/data/'
    f1 = file1[16:]
    f2 = file2[16:]
elif which == 'happy_side':
    suffix = 'happy_side/data/'
    f1 = file1[15:]
    f2 = file2[15:]
elif which == 'happy_back':
    suffix = 'happy_back/data/'
    f1 = file1[15:]
    f2 = file2[15:]

source, target, source_fpfh, target_fpfh = prepare_dataset(suffix+file1+'.ply',suffix+file2+'.ply',voxel_size)

arr1 = source_fpfh.data.T
arr2 = target_fpfh.data.T
nbrs = NN(radius=10,p=33).fit(arr1)

count = 0
mat = np.zeros((arr1.shape[0],arr2.shape[0]))
print(mat.shape)
for i in range(arr1.shape[0]):
    dist,idx = nbrs.radius_neighbors(arr1[i].reshape(1,-1))
    for j in range(idx.shape[0]):
        mat[i,idx[j]] = dist[j]
row_ind, col_ind = LSA(mat)

idx = col_ind

#print(f'{count} out of {arr1.shape[0]} points have at least one neighbour in target point cloud.')