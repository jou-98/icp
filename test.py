import numpy as np
import open3d as o3d 
from icp_2 import *
from utils import *

pcd1,pts1 = o3d_read_pc('bunny/data/bun000.ply')
pcd2,pts2 = o3d_read_pc('bunny/data/bun180.ply')

key1 = o3d.geometry.keypoint.compute_iss_keypoints(pcd1)
key2 = o3d.geometry.keypoint.compute_iss_keypoints(pcd2)
kp1 = np.asarray(key1.points)
kp2 = np.asarray(key2.points)


T,dist,i,_=icp(kp1,kp2)
print(T)
draw_registration_result(pcd1,pcd2,transformation=T,filename='iss_test.ply') 
draw_registration_result(pcd1,pcd2,transformation=None,filename='iss_orig.ply')
       