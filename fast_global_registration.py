import open3d as o3d 
import numpy as np
from utils import *
from time import time
import argparse
from Logger import Logger 
from numpy.linalg import norm as pnorm
from metadata import *


def prepare_dataset(src,dst,voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source,_ = o3d_read_pc(src)
    target,_ = o3d_read_pc(dst)
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def calc_one_fgr(file1,file2,voxel_size=0.001,which='bunny',logger=None):    
    meta_start = time()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
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
    print(f'==================== Testing {f1}.ply against {f2}.ply ====================')

    
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(suffix+file1+'.ply',suffix+file2+'.ply',voxel_size)
    start = time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time() - start))

    logger.record_meta(time()-meta_start)
    T = result_fast.transformation 
    confdir = get_conf_dir(name=which)
    tx1,ty1,tz1,qx1,qy1,qz1,qw1 = get_quat(confdir,file1)
    tx2,ty2,tz2,qx2,qy2,qz2,qw2 = get_quat(confdir,file2)
    t1 = quat_to_mat(tx1,ty1,tz1,qx1,qy1,qz1,qw1)
    t2 = quat_to_mat(tx2,ty2,tz2,qx2,qy2,qz2,qw2)
    reGT = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)
    print(f'Rotation angle between source and target is {round(reGT,3)}')
    mat = np.matmul(trans_init[:3,:3],t1[:3,:3])
    mat = np.matmul(T[:3,:3],mat)
    qx1, qy1, qz1, qw1 = mat_to_quat(mat)
    TE = pnorm(t1[:3,3]-(t2[:3,3]-T[:3,3]), ord=2)
    RE = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
    logger.record_re(RE)
    logger.record_reGT(reGT)
    logger.record_te(TE)

    #draw_registration_result(source, target, filename=which+'/baseline_fgr/'+f1+'_'+f2+'_orig.ply')
    #draw_registration_result(source, target, T,filename=which+'/baseline_fgr/'+f1+'_'+f2+'.ply')
    print(f'============================== End of evaluation ==============================\n\n')
    logger.increment()
    return logger


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bunny', help='dataset to compare')
    parser.add_argument('--voxel_size', type=float, default=0.001, help='size of each voxel')
    FLAGS = parser.parse_args()


    RE_list = []
    TE_list = []
    t_list = []
    log = Logger()

    voxel_size = FLAGS.voxel_size
    which = FLAGS.dataset

    if which == 'bunny':
        for i in range(len(bunny_files)):
            for j in range(len(bunny_files)):
                if i != j:
                    log = calc_one_fgr(bunny_files[i],bunny_files[j],voxel_size=voxel_size,which='bunny',logger=log)
    elif which == 'happy_stand':
        for i in range(len(stand_files)):
            for j in range(len(stand_files)):
                if j != i and np.abs(j-i)<4:
                    log = calc_one_fgr(stand_files[i],stand_files[j],voxel_size=voxel_size,which='happy_stand',logger=log)
    elif which == 'happy_side':
        for i in range(len(side_files)):
            for j in range(len(side_files)):
                if j != i and np.abs(j-i)<4:
                    log = calc_one_fgr(side_files[i],side_files[j],voxel_size=voxel_size,which='happy_side',logger=log)
    elif which == 'happy_back':
        for i in range(len(back_files)):
            for j in range(len(back_files)):
                if j != i and np.abs(j-i)<4:
                    log = calc_one_fgr(back_files[i],back_files[j],voxel_size=voxel_size,which='happy_back',logger=log)
    
    print(f'Results for fast global registration algorithm on {which} dataset.')
    print(f'In total, {log.count} pairs of point clouds are evaluated.')
    print(f'Recall rate is {round(log.recall(),2)}')
    print(f'Average time to compute each pair is {round(log.avg_meta(),3)}s')
    #print(f'Average time to downsample the point cloud is {round(log.avg_sampling(),3)}')
    #print(f'Average time to find correspondence is {round(log.avg_nn(),3)}s')
    #print(f'Average time to compute transformation matrix is {round(log.avg_mat(),3)}s')



