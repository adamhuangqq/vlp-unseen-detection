import  hypertools as hyp
import numpy as np
import csv


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

txt_path = 'F:\object-detection\datasets\VOCdevkit\VOC2007\\原型向量-400张图/ann_feature_emd/'
# save_file = txt_path+'feats.csv'
all_feats = []#np.array([[0]*768])
labels = []
k_class = 20
for i in range(k_class):
    feats = np.loadtxt(txt_path+str(i)+'_match.txt',delimiter=' ')
    all_feats.append(feats[:6,:])#[:4,:])
    labels.extend([i]*6)#feats.shape[0])
    #labels = [i]*feats.shape[0]
    #print(all_feats.shape)
    # print(labels)
    # break
    #writer.write(labels + list(feats))
# feats = np.loadtxt(txt_path+'class_emds.txt',delimiter=' ')[:8,:]
# all_feats.append(feats)#)
# labels.extend([20]*feats.shape[0])
feats = np.loadtxt('F:\object-detection\datasets\VOCdevkit\VOC2007\\原型向量-400张图/10_shot.txt',delimiter=' ')[:k_class,:]
all_feats.append(feats)
labels.extend([20]*feats.shape[0])

#la = [str(i) for i in labels]
#clo = ['b']*20+['r']
#print(clo)

clust = hyp.cluster(all_feats, cluster='KMeans',n_clusters = k_class)
hyp.plot(all_feats, '.', group=labels,  legend=True, ndims=2, reduce='PCA', hue=clust)#labels=la,
plt.legend()
plt.show()
