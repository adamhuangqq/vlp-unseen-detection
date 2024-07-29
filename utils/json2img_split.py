# 本文件是按照json文件标注的数据集分割把图片分割到不同的文件夹中

import json
import numpy as np
import os
import shutil


if __name__ == '__main__':
    # 要转换的json文件
    jsfile_path = 'C:\HQQ\SSOD\MixPL\data\coco\\annotations\instances_val2017.json'
    jsfile = open(jsfile_path, 'r')
    content = jsfile.read()
    all_content = json.loads(content)
    jsfile.close()
    image_list = [i["file_name"][:-4] for i in all_content['images']]
    print(len(image_list))

    img_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\Annotations/'
    # 新划分的保存路径
    new_save_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007/labels/val/'
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)
    for img in image_list:
        shutil.copy(img_path+img+'.xml', new_save_path+img+'.xml')

    # img_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\images/'
    # ann_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\Annotations/'
    # save_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\labels/'
    # parts = ['unlabel', 'train', 'test', 'val']

    # for part in parts:
    #     new_save_path = save_path+part
    #     if not os.path.exists(new_save_path):
    #         os.makedirs(new_save_path)
    #     for img in os.listdir(img_path+part):
    #         shutil.copy(ann_path+img[:-4]+'.xml', new_save_path+'/'+img[:-4]+'.xml')






    
