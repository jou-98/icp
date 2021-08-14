import numpy as np
import open3d as o3d 
from icp_2 import *
from utils import *
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
from numpy.linalg import norm as pnorm


def np_to_p3d(pts):
    plist = pts.tolist()
    p3dlist = []
    for x,y,z in plist:
        pt = POINT3D(x,y,z)
        p3dlist.append(pt)
    return p3dlist

goicp = GoICP()
rNode = ROTNODE()
tNode = TRANSNODE()
which = 'bunny'
file1 = 'bunny/data/bun000.ply'
file2 = 'bunny/data/bun045.ply'
f1 = 'bun000'
f2 = 'bun045'
pcd1,pts1 = o3d_read_pc(file1)
pcd2,pts2 = o3d_read_pc(file2)
"""
c1 = np.mean(pts1,axis=0)
c2 = np.mean(pts2,axis=0)

pts1 -= c1 
pts2 -= c2 
"""

p3d1 = np_to_p3d(pts1)
p3d2 = np_to_p3d(pts2)

rNode.a = -3.1416*0.5
rNode.b = -3.1416*0.5
rNode.c = -3.1416*0.5
rNode.w = 6.2832*0.5
 
tNode.x = -0.15
tNode.y = -0.15
tNode.z = -0.15
tNode.w = 0.40

print(f'Points in pcd1 are within [{np.min(pts1)},{np.max(pts1)}]')
print(f'Points in pcd2 are within [{np.min(pts2)},{np.max(pts2)}]')


goicp.loadModelAndData(len(pts1), p3d1, len(pts2), p3d2)
goicp.setDTSizeAndFactor(300, 2.0)
goicp.setInitNodeRot(rNode)
goicp.setInitNodeTrans(tNode)
goicp.MSEThresh = 0.00001
goicp.trimFraction = 0.00
  
if(goicp.trimFraction < 0.001):
    goicp.doTrim = False

goicp.BuildDT()
goicp.Register()
#print(goicp.optimalRotation()) # A python list of 3x3 is returned with the optimal rotation
#print(goicp.optimalTranslation())# A python list of 1x3 is returned with the optimal translation

T = np.zeros((4,4))
T[:3,:3] = np.array(goicp.optimalRotation())
T[:3,3] = np.array(goicp.optimalTranslation())
T[3,3] = 1
confdir = get_conf_dir(name=which)
tx1,ty1,tz1,qx1,qy1,qz1,qw1 = get_quat(confdir,f1)
tx2,ty2,tz2,qx2,qy2,qz2,qw2 = get_quat(confdir,f2)
t1 = quat_to_mat(tx1,ty1,tz1,qx1,qy1,qz1,qw1)
t2 = quat_to_mat(tx2,ty2,tz2,qx2,qy2,qz2,qw2)
reGT = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)
print(f'Rotation angle between source and target is {round(reGT,3)}, original distance is {pnorm(t1[:3,3]-t2[:3,3], ord=2)}')
mat = np.dot(T[:3,:3],t1[:3,:3])
qx1, qy1, qz1, qw1 = mat_to_quat(mat)
TE = pnorm(t1[:3,3]-(t2[:3,3]-T[:3,3]), ord=2)
RE = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
print(T)

draw_registration_result(pcd1, pcd2, transformation=None, filename='goicp_orig.ply')
draw_registration_result(pcd1, pcd2, transformation=T, filename='goicp.ply')