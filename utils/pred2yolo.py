import os
import numpy as np
import json

if __name__=='__main__':
    pred_file = 'C:\HQQ\SSOD\RTDETR-main/runs/val/r50\predictions.json'

    jsfile = open(pred_file, 'r')
    content = jsfile.read()
    all_content = json.loads(content)
    jsfile.close()
    # print(type(all_content))

    save_path = 'C:\HQQ\SSOD\RTDETR-main/runs/val/r50/pred_yolo/'
    for bbox in all_content:
        print(bbox["image_id"])
        f = open(save_path+bbox["image_id"]+'.txt','a')
        cls = bbox["category_id"]
        conf = bbox["score"]
        box = bbox["bbox"]
        f.write(str(cls)+' '+str(conf)+' '+str(box[0])+' '+str(box[1])+' '+str(box[0]+box[2])+' '+str(box[1]+box[3])+'\n')
        f.close()

    