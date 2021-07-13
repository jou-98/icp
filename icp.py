import open3d as o3d
import numpy as np
import copy
from utils import *
import time
from sklearn.neighbors import NearestNeighbors as NN
from numpy.linalg import norm as pnorm

def correspondence_search(x,y):
    nbrs = NN(n_neighbors=1, algorithm='kd_tree').fit(y)
    dist, idx = nbrs.kneighbors(x)
    return dist.ravel(), idx.ravel()


def compute_mse(x,y,r,t):
    # Normalize the points
    centroid_x = np.mean(x, axis=0)
    centroid_y = np.mean(y, axis=0)
    xx = x - centroid_x 
    yy = y - centroid_y
    # rotate
    xx_rot = np.matmul(xx,r)
    # translation
    xx_trans = xx_rot+t
    # p-norm calculation
    return pnorm_sum(xx_trans,yy)

def compute_matrix(x,y):
    dim = x.shape[1]
    # Normalize the points
    centroid_x = np.mean(x, axis=0)
    centroid_y = np.mean(y, axis=0)
    xx = x - centroid_x 
    yy = y - centroid_y
    #print(f'shape of xx is {xx.shape}, shape of yy is {yy.shape}')
    H = np.matmul(xx.T,yy)
    U, S, V = np.linalg.svd(H)
    # last dimension
    if(np.linalg.det(H) < 0):
        V[-1,:] *= -1
    # Rotation matrix
    R = np.matmul(V.T, U.T)
    # Translation matrix
    t = -np.matmul(R,centroid_x.T) + centroid_y.T
    # Homogeneous metrix
    T = np.identity(dim+1)
    T[:dim,:dim] = R 
    T[:dim,dim] = t

    return T, R, t

def pnorm_sum(x,y):
    dlist = np.apply_along_axis(lambda x:pnorm(x,ord=2),1, x-y)
    return np.sum(dlist)

def compute_d(x,y):
    return pnorm_sum(x,y)/x.shape[0]

def icp(x, y, max_iter=100, threshold=0.3):
    # TODO: change to a better downsampling method
    
    dim = x.shape[1]
    max_points = min(x.shape[0],y.shape[0])
    y = y[:max_points,:]
    x = x[:max_points,:]

    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)

    #print(f'Shape of y is {y.shape}')
    i = 0
    dist = 0
    T = 0
    prev_error = 0

    time_search = 0
    time_matrix = 0
    

    while True:
        i += 1
        start = time.time()
        dist, idx = correspondence_search(src[:dim,:].T, dst[:dim,:].T)
        time_search += time.time() - start
        #print(f'shape of y_new is {y_new.shape}')
        start = time.time()
        T, R, t = compute_matrix(src[:dim,:].T, dst[:dim,idx].T)
        time_matrix += time.time() - start
        #print(f'shape of R is {R.shape} and shape of x is {x.shape}')
        src = np.dot(T, src)
        error = np.mean(dist)
        print(f'Iteration {i}: error = {error}')
        if np.abs(error-prev_error) <= threshold or i == max_iter:
            break 
        prev_error = error
    start = time.time()
    T, R, t = compute_matrix(x, src[:dim,:].T)
    time_matrix += time.time() - start 
    print(f'Time taken to find corresponding points: {round(time_search,3)}s')
    print(f'Time taken to find transformation matrix: {round(time_matrix,3)}s')
    return T, dist, i





if __name__ == "__main__":
    pcd1 = o3d.io.read_point_cloud('bunny/data/bun000.ply',format='ply')
    pcd2 = o3d.io.read_point_cloud('bunny/data/bun270.ply',format='ply')
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    max_points = min(pts1.shape[0],pts2.shape[0])
    pts1 = pts1[:max_points,:]
    pts2 = pts2[:max_points,:]
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    #print(f'Shape of pts1 is {pts1.shape}; Shape of pts2 is {pts2.shape}')
    T, dist, i = icp(pts1, pts2, max_iter=30, threshold=0.00001)
    print(T)
    print(np.mean(dist))
    print(f'Done in {i} iterations')
    draw_registration_result(pcd1, pcd2, T)
    

