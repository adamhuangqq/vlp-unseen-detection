import xml.etree.ElementTree as ET
import os, cv2
import numpy as np
from os import listdir
from os.path import join
import shutil

def count_lines_in_file(file_path):
    """统计单个文件的行数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)

def get_file_list(path):
    """返回文件夹下所有文件和非空文件的列表"""
    file_list = os.listdir(path)
    file_list_fill = []
    for file in file_list:
        if count_lines_in_file(path+file):
            file_list_fill.append(file)
    return file_list, file_list_fill

def delete_file_paste(train_img_path, train_ann_path, new_ann_path, new_img_path):
    """删除掉原来的ann和img，换上新的ann和img"""
    file_list, file_list_fill = get_file_list(new_ann_path)
    print("总共的文件数：%d"%(len(file_list)))
    print("非空的文件数：%d"%(len(file_list_fill)))
    """删除"""
    for file in file_list:
        if os.path.exists(train_ann_path+file):
            os.remove(train_ann_path+file)
            os.remove(train_img_path+file[:-4]+'.jpg')
    """复制"""
    for file in file_list_fill:
        # print(train_ann_path+file)
        shutil.copy(new_ann_path+file,train_ann_path+file)
        shutil.copy(new_img_path+file[:-4]+'.jpg',train_img_path+file[:-4]+'.jpg')

classes = ['aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',]

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    box[0] = max(0,box[0])
    box[1] = max(0,box[1])
    box[2] = min(size[0],box[2])
    box[3] = min(size[1],box[3])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(int_path, yolo_path, img_path):
    ann_list = os.listdir(int_path)
    for file in ann_list:
        img = cv2.imread(img_path+file[:-4]+'.jpg')
        shp = img.shape
        size = [shp[1],shp[0]]
        with open(int_path+file,'r') as ann:
            ann_list = [x.strip() for x in ann.readlines()]
        res = []
        for a in ann_list:
            a = a.split(' ')
            name = a[0]
            if name in classes:
                idx = classes.index(name)
            else:
                continue
            conf = a[1]
            bbox = [int(x) for x in a[2:]]
            # print(bbox)
            # print(size)
            bbox = convert(size, bbox)
            res.append(str(idx) + " " + " ".join([str(a) for a in bbox]))
        #if len(res) != 0:
        with open(yolo_path+file,'w+') as f:
            f.write('\n'.join(res))
        # print(f'{file} convert done!')
    
    
if __name__ == "__main__":
    print(f'Dataset Classes:{classes}')
    img_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\JPEGImages/'
    int_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\pl\\pseudo_label_blip_0.3_0.65/'
    yolo_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\pl\pseudo_label_yolo/'
    
    if not os.path.exists(yolo_path):
        os.makedirs(yolo_path, exist_ok=True)
    
    convert_annotation(int_path, yolo_path, img_path)
    