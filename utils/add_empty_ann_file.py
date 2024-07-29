import os
import shutil

img_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\images\\test/'
ann_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\\voc_labels\\no_difficult\Annotations_allobj_yolo/'
new_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\od_dataset\labels/test/'

img_list = os.listdir(img_path)
ann_list = os.listdir(ann_path)
# for img in img_list:
#     if not os.path.exists(ann_path+img[:-4]+'.txt'):
#         f = open(ann_path+img[:-4]+'.txt','w')
#         f.close()
#     else:
#         continue
for img in img_list:
    shutil.copy(ann_path+img[:-4]+'.txt', new_path+img[:-4]+'.txt')
    
