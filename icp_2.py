import open3d as o3d
import numpy as np
import copy
from utils import *
from time import time
from sklearn.neighbors import NearestNeighbors as NN
from numpy.random import default_rng
from pynndescent import NNDescent as NND
from sampling import *
from Logger import Logger


def correspondence_search(x,y,nbrs=None):
    if nbrs is None:
        nbrs = NN(n_neighbors=1, algorithm='kd_tree').fit(y)
    dist, idx = nbrs.kneighbors(x)
    return dist.ravel(), idx.ravel(), nbrs



def group_icp(g1,g2,iter=30,threshold=0.001):
    pts1 = g1.get_pts()
    pts2 = g2.get_pts()
    labels1 = g1.get_labels()
    labels2 = g2.get_labels()
    vals1 = np.unique(labels1)
    vals2 = np.unique(labels2)
    # Matrix that contains m*n transformation matrices
    transformations = np.zeros((labels1.shape[0],labels2.shape[0],4,4))
    dMat = np.zeros((len(vals1),len(vals2)))
    t = []
    for i in range(vals1.shape[0]):
        idx_i = np.where(labels1==vals1[i])
        c_i = pts1[idx_i]
        for j in range(vals2.shape[0]):
            start = time()
            idx_j = np.where(labels2==vals2[j])
            c_j = pts2[idx_j]
            T,dist,_,_ = icp(c_i,c_j,max_iter=50,threshold=0.00001)
            transformations[i,j] = T
            #print(f'mean(dist) = {np.mean(dist)}')
            dMat[i,j] = np.mean(dist) 
            t.append(time()-start)
    print(f'Total number of pairs: {len(t)}; Avg time taken:{round(np.mean(t),4)}s; Total time taken: {round(np.sum(t),4)}s.')
    return dMat, transformations
            



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

# Carries out ICP algorithm on a pair of point clouds.
# Includes: Correspondence search, 
def first_icp(x, y, idx):
    
    dim = x.shape[1]

    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)
    #print(f'Shape of y is {y.shape}')
    i = 0
    T = 0

    time_search = 0
    count_search = 0
    time_matrix = 0
    count_matrix = 0
    metric = 0
    T, R, t = compute_matrix(src[:dim,:].T, dst[:dim,idx].T)
    time_matrix += time() - start
    count_matrix += 1
    src = np.dot(T, src)
    return T



# Carries out ICP algorithm on a pair of point clouds.
# Includes: Correspondence search, 
def icp(x, y, max_iter=100, threshold=0.00001, logger=None, init=None):
    
    dim = x.shape[1]

    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)

    if init is not None: src = pts_transform(x,init,dim_return=4).T

    #print(f'Shape of y is {y.shape}')
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
        if i % 10 == 0:
            #print(f'Iteration {i}: Error = {error}')
            pass
        if np.abs(error-prev_error) <= threshold or i == max_iter:
            metric = cRMS(src[:dim].T, dst[:dim,idx].T)
            break 
        prev_error = error
    start = time()
    T, R, t = compute_matrix(x, src[:dim,:].T)
    time_matrix += time() - start

    if logger is not None:
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
    print(f'==================== Testing {file1}.ply against {file2}.ply ====================')
    suffix=''
    if which == 'bunny':
        suffix = 'bunny/data/'
    elif which == 'happy_stand':
        suffix = 'happy_stand/'

    
    pcd1 = o3d.io.read_point_cloud(suffix+file1+'.ply',format='ply')
    pcd2 = o3d.io.read_point_cloud(suffix+file2+'.ply',format='ply')
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    min_points = min(pts1.shape[0],pts2.shape[0])
    max_points = max(pts1.shape[0],pts2.shape[0])
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd2.points = o3d.utility.Vector3dVector(pts2)
    start = time()
    if max_points != min_points:
        pts1, pts2 = downsample(pts1,pts2,max_points,min_points,method='fps') # Changed from np_rand
    
    time_sample = time() - start
    print(f'Time taken to downsample the point cloud: {round(time_sample,3)}s')
    logger.record_sampling(time_sample)
    #print(f'Shape of pts1 is {pts1.shape}; Shape of pts2 is {pts2.shape}')
    T, dist, i, logger = icp(pts1, pts2, max_iter=100, threshold=0.0001,logger=logger)
    print(f'Done in {i} iterations')

    confdir = get_conf_dir()
    tx1,ty1,tz1,qx1,qy1,qz1,qw1 = get_quat(confdir,file1)
    tx2,ty2,tz2,qx2,qy2,qz2,qw2 = get_quat(confdir,file2)
    t1 = quat_to_mat(tx1,ty1,tz1,qx1,qy1,qz1,qw1)
    t2 = quat_to_mat(tx2,ty2,tz2,qx2,qy2,qz2,qw2)
    qx1, qy1, qz1, qw1 = mat_to_quat(np.dot(T[:3,:3],t1[:3,:3]))
    TE = pnorm(t1[:3,3]-(t2[:3,3]-T[:3,3]), ord=2)
    RE = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
    logger.record_re(RE)
    logger.record_te(TE)
    draw_registration_result(pcd1, pcd2, transformation=None, filename='bunny/results/'+file1+'_'+file2+'_orig.ply')
    draw_registration_result(pcd1, pcd2, T, filename='bunny/results/'+file1+'_'+file2+'.ply')
    print(f'============================== End of evaluation ==============================\n\n')
    logger.increment()
    return logger

if __name__ == "__main__":
    pass
    """
    bunny_files =   ['bun000','bun045','bun090','bun180','bun270',
                    'bun315','chin','ear_back','top2','top3']
    RE_list = []
    TE_list = []
    t_list = []
    log = Logger()
    for i in range(len(bunny_files)):
        for j in range(len(bunny_files)):
            if i < j:
                log = calc_one_icp(bunny_files[i],bunny_files[j],log)
    

    print(f'Recall rate is {round(log.recall(),2)}')
    print(f'Average time to compute each pair is {round(log.avg_all(),3)}s')
    print(f'Average time to downsample the point cloud is {round(log.avg_sampling(),3)}')
    print(f'Average time to find correspondence is {round(log.avg_nn(),3)}s')
    print(f'Average time to compute transformation matrix is {round(log.avg_mat(),3)}s')

    """