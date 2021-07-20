import numpy as np
import open3d as o3d
import random
import copy
from numpy.random import default_rng

BIN_DIR = './bin/'
LABELs_DIR = './labels/'

def sampling(arr1,arr2,max_points,min_points,method='default'):
    pts1 = copy.deepcopy(arr1)
    pts2 = copy.deepcopy(arr2)
    if method == 'default':
        pts1 = pts1[:min_points,:] if pts1.shape[0] >= pts2.shape[0] else pts1
        pts2 = pts2[:min_points,:] if pts2.shape[0] > pts1.shape[0] else pts2
        return pts1,pts2
    if method == 'farthest_point':
        pass
    if method == 'np_rand':
        rng = default_rng()
        idx = rng.choice(max_points,size=min_points,replace=False)
    elif method == 'py_rand':
        idx = random.sample(range(max_points),min_points)
    if pts1.shape[0] > pts2.shape[0]: pts1 = pts1[idx,:]
    if pts1.shape[0] < pts2.shape[0]: pts2 = pts2[idx,:]
    return pts1, pts2




# Computes the coordinate Root-Mean-Squared error between the two
# point clouds.
def cRMS(src, dst):
    diff = src-dst
    root_squared = np.apply_along_axis(lambda x:x[0]**2+x[1]**2+x[2]**2 ,axis=1,arr=diff)
    return np.mean(root_squared)


def render_color(pc,label,ply_path='./00_001000_colored.ply'):
    labels = np.unique(label)
    colors = dict()
    for i in labels:
        color = list(np.random.choice(range(100), size=3)/100)
        colors[i] = color
    rgb = np.zeros((pc.shape[0],3))

    for i in labels:
        label_pos = np.argwhere(np.any(label==i, axis=1))#label[label==i]
        if not i in CLASSES:
            continue
        rgb[label_pos] = colors[i]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(ply_path, pcd) 
    
    
    

# Reads a .bin file into numpy array
def read_pc_bin(path='./dataset/sequences/00/velodyne/001000.bin',dim=4):
    scan = np.fromfile(path, dtype=np.float32)
    scan = scan.reshape((-1, dim))     
    return scan[:,:3]

def save_pc(array,ply_path='./00_001000.ply'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array[:,:3])
    o3d.io.write_point_cloud(ply_path, pcd) 


# Reads a .label file into numpy array
def read_label(path='./labels/00_000143.label',dim=2):
    labels = np.fromfile(path, dtype=np.uint16).reshape((-1,dim))
    labels = labels[:,0]
    #print(np.unique(labels,return_counts=True))
    return labels


# Renders a point cloud in black and white, reads either bin file or ply file 
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

# Draws the result of transformation matrix. Takes two open3d point clouds
# as input, not arrays
def draw_registration_result(source, target, transformation, filename='a+b.ply'):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    rgb = np.concatenate((np.asarray(source_temp.colors),np.asarray(target_temp.colors)),axis=0)
    pts = np.concatenate((np.asarray(source_temp.points),np.asarray(target_temp.points)),axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(filename, pcd)
    """
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    """