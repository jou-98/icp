import numpy as np
import open3d as o3d
import copy
from numpy.random import default_rng
import csv
from math import sqrt


BIN_DIR = './bin/'
LABELs_DIR = './labels/'

def get_conf_dir(name='bunny'):
    if name == 'bunny': return 'bunny/data/bun.conf'
    if name == 'happy_stand': return 'happy_stand/data/happyStandRight.conf'
    if name == 'happy_side': return 'happy_side/data/happySideRight.conf'
    if name == 'happy_back': return 'happy_back/data/happyBackRight.conf'

def get_quat(dir,fname):
    buf = []
    with open(dir, newline='') as config:
        reader = csv.reader(config,delimiter=' ')
        for row in reader:
            if len(row) != 0 and row[1] == fname+'.ply':
                buf = np.asarray(row[2:9],dtype=np.float64)
    tx,ty,tz,qx,qy,qz,qw= buf
    return tx,ty,tz,qx,qy,qz,qw


# Computes the coordinate Root-Mean-Squared error between the two
# point clouds.
def cRMS(src, dst):
    diff = src-dst
    if np.max(diff)==np.min(diff): print(f'src is equal to dst!!')
    std = np.std(diff)
    squared = np.apply_along_axis(lambda x:(x[0]**2+x[1]**2+x[2]**2) ,axis=1,arr=diff)
    mean_squared = np.mean(squared)
    #print(f'Max diff is {np.max(diff)} and min diff is {np.min(diff)}')
    #print(f'Standard deviation is {std}')
    return np.sqrt(mean_squared) / (np.max(diff)-np.min(diff))


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
    
def pts_transform(pts,t,dim_return=3):
    if pts.shape[1] == 3:
        buf = np.ones((pts.shape[0],pts.shape[1]+1))
        buf[:,:3] = pts
        pts = buf
    pts_new = np.matmul(t,pts.T).T
    if dim_return == 3:
        return pts_new[:,:3]
    else:
        return pts_new

def o3d_read_pc(path,ext='ply'):
    pcd = o3d.io.read_point_cloud(path,format=ext)
    pts = np.asarray(pcd.points)
    return pcd, pts

def o3d_instance(pts,rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None: pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def o3d_save_pc(pts,path,colors=None,pcd_format=False):
    if pcd_format:
        pcd = pts
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None: pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


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
def draw_registration_result(source, target, transformation=None, filename='a+b.ply'):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    if transformation is not None: source_temp=source_temp.transform(transformation)

    rgb = np.concatenate((np.asarray(source_temp.colors),np.asarray(target_temp.colors)),axis=0)
    pts = np.concatenate((np.asarray(source_temp.points),np.asarray(target_temp.points)),axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(filename, pcd)


def quat_to_mat(tx,ty,tz,qx,qy,qz,qw):
    transformation = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
    transformation[0,0] = 1 - 2 * qy * qy - 2 * qz * qz
    transformation[0,1] = 2 * qx * qy - 2 * qz * qw
    transformation[0,2] = 2 * qx * qz + 2 * qy * qw 
    transformation[1,0] = 2 * qx * qy + 2 * qz * qw 
    transformation[1,1] = 1 - 2 * qx * qx - 2 * qz * qz
    transformation[1,2] = 2 * qy * qz - 2 * qx * qw 
    transformation[2,0] = 2 * qx * qz - 2 * qy * qw 
    transformation[2,1] = 2 * qy * qz + 2 * qx * qw 
    transformation[2,2] = 1 - 2 * qx * qx - 2 * qy * qy 
    transformation[0,3] = tx
    transformation[1,3] = ty
    transformation[2,3] = tz
    return transformation

def mat_to_quat(mat):
    qw = sqrt(1.0 + mat[0,0] + mat[1,1] + mat[2,2]) / 2.0
    qx = (mat[2,1]-mat[1,2]) / (4.0 * qw)
    qy = (mat[0,2]-mat[2,0]) / (4.0 * qw)
    qz = (mat[1,0]-mat[0,1]) / (4.0 * qw)
    # print(f'w^2 + x^2 + y^2 + z^2 = {qx*qx+qy*qy+qz*qz+qw*qw}')
    return qx, qy, qz, qw


def rotation_error(p1,p2,p3,p4,q1,q2,q3,q4):
    cos = np.abs(p1*q1 + p2*q2 + p3*q3 + p4*q4)
    if cos > 1 and cos < 1.001: cos=1.00
    return np.rad2deg(2*np.arccos(cos))