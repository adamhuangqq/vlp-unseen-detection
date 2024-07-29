from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
import numpy as np
import glob
from data.dataloader_file import BLIPDataset, frcnn_dataset_collate
from data.dataloader_file1p5 import BLIPDataset1p5, frcnn_dataset_collate1p5
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-------------use %s-----------'%(device))
    model_name = 'RN50'
    jit_model, transform = clip.load(model_name, device=device)
    py_model, _ = clip.load(model_name, device=device, jit=False)
    img = Image.open("CLIP.png")

    class_file = 'model_data/voc_classes.txt'
    with open(class_file,'r') as f:
        class_list = f.readlines()
        class_list =[x.split('\n')[0] for x in class_list]
    caption = ['a photo of a '+ cls + ' in the scene.' for cls in class_list]
    print('text: %s' %caption)
    text = clip.tokenize(caption).to(device)

    img_path = 'C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\JPEGImages/'
    ann_folder = 'C:\HQQ\SSOD\MixPL/test_results\mixpl_tood\preds2blip/'
    save_path = 'C:\HQQ\SSOD\MixPL/test_results\mixpl_tood\pl_clip_0.3_0.8/'

    data = BLIPDataset(img_path, ann_folder)
    gen = DataLoader(data, shuffle = False, batch_size = 1, num_workers = 8, pin_memory=True,
                        drop_last=True, collate_fn=frcnn_dataset_collate
                        )
    data2 = BLIPDataset1p5(img_path, ann_folder)
    gen2 = DataLoader(data2, shuffle = False, batch_size = 1, num_workers = 8, pin_memory=True,
                        drop_last=True, collate_fn=frcnn_dataset_collate
                        )
    print('dataset done!')

    def file_write(obj_list, conf, boxes, save_path, file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(save_path+file_name, 'w')
        for i in range(len(obj_list)):
            if obj_list[i] != 'background':
                f.write(obj_list[i]+' ')
                f.write(str(conf[i])+' ')
                f.write(str(boxes[i][0])+' '+str(boxes[i][1])+' '+str(boxes[i][2])+' '+str(boxes[i][3])+'\n')
        f.close()

    length = len(os.listdir(ann_folder))
    threhold = 0.8

    with tqdm(total=length, desc=f'num 1', postfix=dict, mininterval=0.3) as pbar:
        for iteration, (batch, batch2) in enumerate(zip(gen, gen2)):
            crops_batch, confs_batch, boxes_batch, file_name_list = batch[0], batch[1], batch[2], batch[3]
            crops_batch2, confs_batch2, boxes_batch2, file_name_list2 = batch2[0], batch2[1], batch2[2], batch2[3]
            for i in range(crops_batch.shape[0]):
                obj_list = []
                file_name = file_name_list[i]
                conf = confs_batch[i]
                boxes = boxes_batch[i]
                crop = crops_batch[i].to(device)
                crop2 = crops_batch2[i].to(device)
                if crop.shape[0]==0:
                    file_write(obj_list, conf, boxes, save_path, file_name)
                    continue
                with torch.no_grad():
                    image_feat, _ = jit_model(crop, text)
                    image_feat2, _ = jit_model(crop2, text)
                image_feat = torch.add(image_feat,image_feat2)/2.0
                itc_score = image_feat.softmax(dim=-1).cpu().numpy()
                image_feat = image_feat.cpu().numpy().astype(np.float32)


                argmax_itc = itc_score.argmax(axis=1)
                argmax_itc = argmax_itc.tolist()

                for i in range(len(argmax_itc)):
                    # print(itc_score[i,argmax_itc[i]])
                    if itc_score[i,argmax_itc[i]]+conf[i] >= threhold:
                        obj_list.append(class_list[argmax_itc[i]])
                    else:
                        obj_list.append('background')
                file_write(obj_list, conf.cpu().numpy(), boxes, save_path, file_name)
            pbar.set_postfix(**{'file_name': file_name
                                #'num':num
                                })
            pbar.update(1)
            #break
            




