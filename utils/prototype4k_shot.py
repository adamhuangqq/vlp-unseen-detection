# 得到k-shot的原型向量

import os
import numpy as np

if __name__ == '__main__':
    k_shot = 10
    save_path = 'F:\object-detection\datasets\VOCdevkit\VOC2007\原型向量-400张图/{}_shot_feat.txt'.format(k_shot)
    file_path = 'F:\object-detection\datasets\VOCdevkit\VOC2007\原型向量-400张图\\ann_feature_emd/'
    class_num = 20
    f = open(save_path, 'w')
    for i in range(class_num):
        file_name = file_path+str(i)+'.txt'
        feats = np.loadtxt(file_name, delimiter=' ')[:k_shot,:]
        prototype = np.average(feats, axis=0)
        #print(type(feats))
        prototype_str = ' '.join([str(item) for item in prototype.tolist()])
        f.write(prototype_str)
        f.write('\n')

