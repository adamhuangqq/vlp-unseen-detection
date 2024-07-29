import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
import shutil


def get_ground_truth(gt_path, map_out_path):
    print("Get ground truth result.")
    image_ids = [k[:-4] for k in os.listdir(gt_path)]
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(gt_path+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    #print(image_id)
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")
if __name__=='__main__':
    class_path = 'model_data/voc_classes.txt'
    class_names,_ = get_classes(class_path)
    # gt_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\labels/val/'
    # map_out_path = 'C:\HQQ\SSOD\MixPL\output/fcos-50%/val/test/'
    # get_ground_truth(gt_path, map_out_path)
    path = 'C:\HQQ\SSOD\MixPL\output/fcos-50%/val/test/'
    #get_coco_map(class_names, path)
    temp_map = get_map(0.5, False, 0.5, path = path)
    shutil.rmtree(path+'results/')
    print(temp_map)

