# 本文件是将mixpl中的预测结果转换为yolo形式的标签，
# 以方便后面BLIP的使用（即和之前的代码接轨了）

import json
import numpy as np
import os
import shutil


if __name__ == '__main__':
    jsfile_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\preds/'
    new_save_path = 'C:\HQQ\SSOD\MixPL\\test_results\mixpl_tood\preds2blip/'
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)
    label_list = os.listdir(jsfile_path)
    for label in label_list:
        jsfile = open(jsfile_path+label, 'r')
        content = jsfile.read()
        all_content = json.loads(content)
        jsfile.close()
        file_name = new_save_path+label[:-5]+'.txt'
        f = open(file_name, 'w')
        for i in range(len(all_content["labels"])):
            f.write(str(all_content["labels"][i])+' '+ str(all_content["scores"][i])+' ')
            bbox = [str(int(k)) for k in all_content["bboxes"][i]]
            f.write(bbox[0]+' '+bbox[1]+' '+bbox[2]+' '+bbox[3]+'\n')
        f.close()
    
        


            
            

