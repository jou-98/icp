from pyflann import *
import numpy as np

dataset = np.random.random((2000,33))
testset = np.random.random((2000,33))
set_distance_type("euclidean")
flann = FLANN()
params = flann.build_index(dataset, algorithm='kdtree', trees=4)
k_nearest = 200
result, dists = flann.nn_index(testset,k_nearest,checks=params["checks"])

print(dists[:3])