import open3d as o3d
import numpy as np
import copy
from utils import *
from time import time
from sklearn.neighbors import NearestNeighbors as NN
from numpy.random import default_rng
from sampling import *
from Logger import Logger
from matplotlib import pyplot as plt
from metadata import *


def correspondence_search(x,y,nbrs=None):
    if nbrs is None: 
        nbrs = NN(n_neighbors=1, algorithm='kd_tree').fit(y)
    dist, idx = nbrs.kneighbors(x)
    return dist.ravel(), idx.ravel(), nbrs


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

def icp(x, y, max_iter=1000, threshold=0.00001, logger=None):    
    dim = x.shape[1]

    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)

    d_trend = []
    i = 0
    dist = 0
    T = 0
    prev_error = 0

    time_search = 0
    count_search = 0
    time_matrix = 0
    count_matrix = 0
    metric = 0
    nbrs = None
    while True:
        i += 1
        start = time()
        if nbrs is None: dist, idx, nbrs = correspondence_search(src[:dim,:].T, dst[:dim,:].T)
        if nbrs is not None: dist, idx, _ = correspondence_search(src[:dim,:].T, dst[:dim,:].T, nbrs)
        time_search += time() - start
        count_search += 1
        start = time()
        T, R, t = compute_matrix(src[:dim,:].T, dst[:dim,idx].T)
        time_matrix += time() - start
        count_matrix += 1
        src = np.dot(T, src)
        error = np.mean(dist)
        d_trend.append(error)
        if i % 10 == 0:
            pass
            #print(f'Iteration {i}: error = {error.round(5)}')
        if np.abs(error-prev_error) <= threshold or i == max_iter:
            metric = cRMS(src[:dim].T, dst[:dim,idx].T)
            break 
        prev_error = error
    start = time()
    T, R, t = compute_matrix(x, src[:dim,:].T)
    time_matrix += time() - start 
    logger.record_nn(time_search)
    logger.record_mat(time_matrix)


    print(f'Time taken to find corresponding points: {round(time_search,3)}s')
    print(f'Average time for each correspondence search {round(time_search/count_search,3)}s')
    print(f'Time taken to find transformation matrix: {round(time_matrix,3)}s')
    print(f'Average time for each matrix computation {round(time_matrix/count_matrix,3)}s')
    print(f'Total time taken: {round(time_search+time_matrix,3)}')
    print(f'cRMS = {round(metric,3)}')
    return T, dist, i, logger

def calc_one_icp(file1, file2, logger=None, which='bunny'):
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
    print(f'==================== Testing {f1} against {f2} ====================')

    
    pcd1 = o3d.io.read_point_cloud(suffix+file1+'.ply',format='ply')
    pcd2 = o3d.io.read_point_cloud(suffix+file2+'.ply',format='ply')
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    min_points = min(pts1.shape[0],pts2.shape[0])
    max_points = max(pts1.shape[0],pts2.shape[0])
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    start = time()
    if pts1.shape[0] != pts2.shape[0]:
        pts1, pts2 = downsample(pts1,pts2,method='fps') # Changed from np_rand
    print(f'Shape of pts1 is {pts1.shape}')
    print(f'Shape of pts2 is {pts2.shape}')

    time_sample = time() - start
    print(f'Time taken to downsample the point cloud: {round(time_sample,3)}s')
    logger.record_sampling(time_sample)
    #print(f'Shape of pts1 is {pts1.shape}; Shape of pts2 is {pts2.shape}')
    T, dist, i, logger = icp(pts1, pts2, max_iter=200, threshold=0.00001,logger=logger)
    print(f'Done in {i} iterations')
    print(T)
    confdir = get_conf_dir(name=which)
    tx1,ty1,tz1,qx1,qy1,qz1,qw1 = get_quat(confdir,file1)
    tx2,ty2,tz2,qx2,qy2,qz2,qw2 = get_quat(confdir,file2)
    t1 = quat_to_mat(tx1,ty1,tz1,qx1,qy1,qz1,qw1)
    t2 = quat_to_mat(tx2,ty2,tz2,qx2,qy2,qz2,qw2)
    reGT = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)
    print(f'Rotation angle between source and target is {round(reGT,3)}')
    # print(f'Extra RE = {rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)}')
    qx1, qy1, qz1, qw1 = mat_to_quat(np.dot(T[:3,:3],t1[:3,:3]))
    TE = pnorm(t1[:3,3]-(t2[:3,3]-T[:3,3]), ord=2)
    RE = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
    #print(f'Extra RE = {rotation_error()}')
    logger.record_re(RE)
    logger.record_reGT(reGT)
    logger.record_te(TE)
    #draw_registration_result(pcd1, pcd2, t2, filename=file1+'_'+file2+'ex.ply')
    draw_registration_result(pcd1, pcd2, transformation=None, filename=which+'/baseline_vanilla/'+f1+'_'+f2+'_orig.ply')
    draw_registration_result(pcd1, pcd2, T, filename=which+'/baseline_vanilla/'+f1+'_'+f2+'.ply')
    print(f'============================== End of evaluation ==============================\n\n')
    logger.increment()
    return logger

if __name__ == "__main__":

    RE_list = []
    TE_list = []
    t_list = []
    log = Logger()
    
    which = dataset[0]
    if which == dataset[0]:
        for i in range(len(bunny_files)):
            for j in range(len(bunny_files)):
                if i < j:
                    log = calc_one_icp(bunny_files[i],bunny_files[j],log,which='bunny')
    elif which == dataset[1]:
        for i in range(len(stand_files)):
            for j in range(len(stand_files)):
                if i != j and np.abs(j - i) < 4:
                    log = calc_one_icp(stand_files[i],stand_files[j],log,which='happy_stand')
    elif which == dataset[2]:
        for i in range(len(side_files)):
            for j in range(len(side_files)):
                if j > i:
                    log = calc_one_icp(side_files[i],side_files[j],log,which='happy_side')
    elif which == dataset[3]:
        for i in range(len(side_files)):
            for j in range(len(side_files)):
                if j > i:
                    log = calc_one_icp(back_files[i],back_files[j],log,which='happy_back')
        
    """
    plt.scatter(log.reGT,log.re)
    print(f'reGT={log.reGT}; re={log.re}')
    plt.ylabel('Rotation Error')
    plt.xlabel('Initial Difference')
    plt.title('Initial angle vs. RE for Standing Buddha')
    plt.xlim([0,190])
    plt.ylim([0,190])
    plt.axhline(y=15)
    plt.legend()
    plt.savefig('re_reGT_stand')
    """
    print(f'In total, {log.count} pairs of point clouds are evaluated.')
    print(f'Recall rate is {round(log.recall(),2)}')
    print(f'Average time to compute each pair is {round(log.avg_all(),3)}s')
    print(f'Average time to downsample the point cloud is {round(log.avg_sampling(),3)}')
    print(f'Average time to find correspondence is {round(log.avg_nn(),3)}s')
    print(f'Average time to compute transformation matrix is {round(log.avg_mat(),3)}s')

