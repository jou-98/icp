from sklearn.cluster import KMeans
import numpy as np
from utils import *
from time import time
from random import random
import sampling
from graph import *
from icp_2 import *
from math import exp
from scipy.optimize import quadratic_assignment as QAP
from scipy.optimize import linear_sum_assignment as LSA



nC = 20 # Number of clusters
rgb_map1 = dict()
rgb_map2 = dict()
downsample = True
meta_start = time()
img1 = 'bun000'
img2 = 'bun090'
file1 = 'bunny/data/'+img1+'.ply'
file2 = 'bunny/data/'+img2+'.ply'
pcd1, pts1 = o3d_read_pc(file1)
pcd2, pts2 = o3d_read_pc(file2)
if downsample:
    pts1 = sampling.fps(pcd1, num_points=15000)
    pts2 = sampling.fps(pcd2, num_points=15000)
print(f'Downsampling done.')
start = time()
g1 = Graph(pts1,nC)
g2 = Graph(pts2,nC)
labels1 = g1.get_labels()
labels2 = g2.get_labels()



rgb1 = np.zeros((labels1.shape[0],3))
rgb2 = np.zeros((labels2.shape[0],3))

for i in np.unique(labels1):
    rgb_map1[i] = [random(),random(),random()]
for i in np.unique(labels2):
    rgb_map2[i] = [random(),random(),random()]

for i in rgb_map1:
    rgb1[labels1==i] = rgb_map1[i]
for i in rgb_map2:
    rgb2[labels2==i] = rgb_map2[i]


o3d_save_pc(pts1, img1+f'_cluster{nC}.ply',rgb1)
print(f'pcd1 saved as {img1}_cluster{nC}.ply')
o3d_save_pc(pts2, img2+f'_cluster{nC}.ply',rgb2)
print(f'pcd2 saved as {img2}_cluster{nC}.ply')


dV, mT = group_icp(g1,g2)
#print(dV[5,:])
t = time()
dE = edge_dist(g1,g2,mT)
print(f'Computing edge distances took {time()-t}s.')

aff = affinity_matrix(g1,g2,dV,dE)

# idx = qap_compute(aff)

row_idx,col_idx = LSA(cost)

init = first_icp(g1.centres,g2.centres,idx)

T,dist,i,_ = icp(pts1,pts2,max_iter=100,threshold=0.0001,logger=None,init=init)

draw_registration_result(o3d_instance(pts1), o3d_instance(pts2), transformation=T, filename='test_gm.ply')


print(f'Whole process takes {time()-meta_start}s.')
