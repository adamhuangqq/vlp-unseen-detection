import os
import shutil

if __name__ == '__main__':
    voc_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007/'
    new_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007/yolo_split/'
    split_file = voc_path+'ImageSets\Main/'
    img_path = voc_path+'JPEGImages/'
    label_path = voc_path+'Annotations/'
    new_img_path = voc_path+'images/'
    new_label_path = voc_path+'labels/'
    s_list = ['train', 'val', 'test']
    for name in s_list:
        with open(split_file+name+'.txt','r') as f:
            file_list = f.readlines()
            file_list = [x.split()[0] for x in file_list]
        if not os.path.exists(new_img_path+name+'/'):
            os.makedirs(new_img_path+name+'/')
        if not os.path.exists(new_label_path+name+'/'):
            os.makedirs(new_label_path+name+'/')
        for file in file_list:
            shutil.copy(img_path+file+'.jpg',new_img_path+name+'/'+file+'.jpg')
            shutil.copy(label_path+file+'.xml',new_label_path+name+'/'+file+'.xml')

    # ori_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\labels_xml/'
    # txt_path = 'C:\HQQ\SSOD\RTDETR-main\dataset/voc_labels/no_difficult\Annotations_1obj_yolo/'
    # new_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\labels/'
    # sub_dir  = os.listdir(ori_path)
    # for dir in sub_dir:
    #     if not os.path.exists(new_path+dir+"/"):
    #         os.makedirs(new_path+dir+"/")
    #     for file in os.listdir(ori_path+dir):
    #         shutil.copy(txt_path+file[:-4]+'.txt',new_path+dir+"/"+file[:-4]+'.txt')


        



