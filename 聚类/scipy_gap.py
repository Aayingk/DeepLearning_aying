import scipy
import scipy.cluster.vq
import scipy.spatial.distance
import numpy as np
import numpy
import matplotlib.pyplot as plt
import time
EuclDist = scipy.spatial.distance.euclidean

def gapStat(data, resf=None, nrefs=10, ks=range(1,10)):
    '''
    Gap statistics
    '''
    start = time.perf_counter()
    # MC
    shape = data.shape
    if resf == None:
        x_max = data.max(axis=0)
        x_min = data.min(axis=0)
        dists = np.matrix(np.diag(x_max-x_min))
        rands = np.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+x_min
    else:
        rands = resf
    gaps = np.zeros((len(ks),))
    gapDiff = np.zeros(len(ks)-1,)
    sdk = np.zeros(len(ks),)
    for (i,k) in enumerate(ks):
        print("data",data.shape)
        (cluster_mean, cluster_res) = scipy.cluster.vq.kmeans(data, k)
        print("ssssssss",cluster_mean)
        print("gggggggggg",cluster_res)
        print(data[1,:])
        print(cluster_mean[cluster_res[1],:])
        Wk = sum([EuclDist(data[m,:], cluster_mean[cluster_res[m],:]) for m in range(shape[0])])
        WkRef = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
            WkRef[j] = sum([EuclDist(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
        gaps[i] = numpy.log(numpy.mean(WkRef))-numpy.log(Wk)
        sdk[i] = np.sqrt((1.0+nrefs)/nrefs)*np.std(numpy.log(WkRef))

        if i > 0:
            gapDiff[i-1] = gaps[i-1] - gaps[i] + sdk[i]
    end = time.perf_counter()
    print(gapDiff)
    print("running time:%s Seconds" % (end - start))
    return gaps, gapDiff

mean = (1, 2)
cov = [[1, 0], [0, 1]]
Nf = 1000
dat1 = np.zeros((3000,2))
dat1[0:1000,:] = numpy.random.multivariate_normal(mean, cov, 1000)
mean = [5, 6]
dat1[1000:2000,:] = numpy.random.multivariate_normal(mean, cov, 1000)
mean = [3, -7]
dat1[2000:3000,:] = numpy.random.multivariate_normal(mean, cov, 1000)


gaps,gapsDiff = gapStat(dat1)
f, (a1,a2) = plt.subplots(2,1)
a1.plot(gaps, 'g-o')
a2.bar(np.arange(len(gapsDiff)),gapsDiff)
f.show()