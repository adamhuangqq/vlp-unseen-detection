import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os
import shutil
from yoloint2yolo import delete_file_paste, convert_annotation



train_img_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\od_dataset\images\\train/'
train_ann_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\od_dataset\labels\\train/'
new_ann_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\pl\pseudo_label_yolo/'
new_img_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\images\\val/'

img_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\JPEGImages/'
int_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\pl\\pl_clip_0.3_0.8/'
yolo_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\pl\pseudo_label_yolo/'


if __name__ == '__main__':
    convert_annotation(int_path, yolo_path, img_path)
    delete_file_paste(train_img_path, train_ann_path, new_ann_path, new_img_path)
    print('-------------Dataset done!-------------')

    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml') 
    model.load('weights/rtdetr-r18.pt') # loading pretrain weights
    model.train(data='dataset/data_od.yaml',
                cache=False,
                imgsz=640,
                epochs=120,
                batch=8,
                workers=8,
                device='0',
                # resume='last.pt', # last.pt path
                project='runs/train/r18',
                name='ori-pretrained-clip_0.3_0.8',
                )