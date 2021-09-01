import stat

import scipy.spatial.distance
import numpy as np
import pandas
import matplotlib.pyplot as plt


EuclDist = scipy.spatial.distance.euclidean


mean = (1, 2)
cov = [[1, 0], [0, 1]]
#np.random.multivariate_normal(1.1, [[0,1],[1,0]])
Nf = 1000
dat1 = np.zeros((3000,2))
# 定义 3 类数据，并且拼接
dat1[0:1000,:] = np.random.multivariate_normal(mean, cov, 1000)
mean = [5, 6]
dat1[1000:2000,:] = np.random.multivariate_normal(mean, cov, 1000)
mean = [3, -7]
dat1[2000:3000,:] = np.random.multivariate_normal(mean, cov, 1000)
plt.plot(dat1[::,0], dat1[::,1], 'b.', linewidth=1)

