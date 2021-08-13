# Graph data structure for nearest neighbour search
# Written by Yingchen Nie
import numpy as np 
from numpy.linalg import norm as pnorm
from scipy.optimize import quadratic_assignment as QAP
from sklearn.cluster import KMeans
from math import exp
from time import time
import open3d as o3d

# Converts cij to i1 and j1, for the purpose of 
# affinity matrix construction
def aff_get_ij(c,Nt):
    return int((c - c % Nt)/Nt), c % Nt

# Solves the QAP problem for matrix permutation
def qap_compute(aff):
    A = np.identity(aff.shape[0])
    B = aff 
    res = QAP(A,B,options={'maximize':True})
    return res.col_ind

def idx_convert(idx):
    pass

# Computes the edge distances between two graphs
def edge_dist(g1,g2,transformation):
    p1 = g1.get_pts()
    l1 = g1.get_labels()
    p2 = g2.get_pts()
    l2 = g2.get_labels()
    dist = np.zeros((g1.getNLabel(),g2.getNLabel(),g1.getNLabel(),g2.getNLabel()))
    # i2 at outmost loop to avoid redundant assignment of i2
    for i2 in range(g1.getNLabel()): 
        idx = np.where(l1==l1[i2])
        p_i2 = p1[idx]
        #print(f'i2 = {i2}')
        for i1 in range(g1.getNLabel()):
            for j1 in range(g2.getNLabel()):
                for j2 in range(g2.getNLabel()):
                    if i1 == i2 or j1 == j2: continue
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(p_i2)
                    pcd = pcd.transform(transformation[i1,j1])
                    t1 = np.asarray(pcd.points)
                    pcd.points = o3d.utility.Vector3dVector(p_i2)
                    pcd.transform(transformation[i2,j2])
                    t2 = np.asarray(pcd.points)
                    dist[i1,j1,i2,j2] = pnorm(t2-t1,ord=2)
    return dist

# Computes the affinity matrix
def affinity_matrix(g1,g2,dV,dE):
    Nt = g2.getNLabel()
    NsNt = g1.getNLabel()*g2.getNLabel()
    mat = np.zeros((NsNt, NsNt))
    for c1 in range(NsNt):
        for c2 in range(NsNt):
            i1,j1 = aff_get_ij(c1,Nt)
            i2,j2 = aff_get_ij(c2,Nt)
            # vertex_sim(Vij, Uij), v from g1 and u from g2
            if i1 == i2 and j1 == j2: 
                mat[c1,c2] = vertex_sim(i1,j1,dV,lv=0.5)
            # edge_sim((vi1,vi2),(uj1,uj2)), vi1 vi2 from g1 and uj1 uj2 from g2
            elif i1 != i2 and j1 != j2: 
                mat[c1,c2] = edge_sim(i1,j1,i2,j2,dE,le=0.5)
            else:
                mat[c1,c2] = 0
    return mat

# Similarity function for edges
def edge_sim(i1,j1,i2,j2,dist,le=0.5):
    return exp(-le * dist[i1,j1,i2,j2])

# Similarity function for vertices
def vertex_sim(v1,v2,dist,lv=0.5):
    return exp(-lv * dist[v1,v2])

class Graph:
    def __init__(self,pts,k=100):
        start = time()
        self.adjacency_list = []
        self.adjacency_matrix = np.zeros((pts.shape[0],pts.shape[0]))
        self.pts = pts
        self.affinity = []
        """
        for i in range(pts.shape[0]):
            for j in range(pts.shape[0]):
                if i <= j: continue
                self.adjacency_matrix[i,j] = pnorm(pts[i]-pts[j],ord=2)
        """
        self.model = KMeans(n_clusters=k,algorithm='elkan').fit(pts)
        self.centres = self.model.cluster_centers_
        self.labels = self.model.predict(pts)
        print(f'Graph Initialized. Time Taken: {round(time()-start,3)}s.')

    def getNLabel(self):
        return np.unique(self.labels).shape[0]

    def has_edge(self,v1,v2):
        return self.adjacency_matrix[v1,v2] != -1

    def getN(self):
        return self.adjacency_matrix.shape[0]

    def get_pts(self,copy=True):
        if copy: return np.copy(self.pts)
        return self.pts
    
    def get_labels(self,copy=True):
        if copy: return np.copy(self.labels)
        return self.labels



    def n_step(self,n):
        pass
    

