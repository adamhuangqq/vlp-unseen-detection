from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
from IPython.display import display
import numpy as np
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('-------------use %s-----------'%(device))
def load_demo_image(image_size,device,image_path):
    img_url =  image_path
    raw_image = Image.open(img_url).convert('RGB')   

    w,h = raw_image.size
    #display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

voc_classes = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'dining table',
        'dog',
        'horse',
        'motorbike',
        'person.',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    ]


from models.blip_itm import blip_itm

image_size = 384
image_path = 'F:\object-detection\datasets\VOCdevkit\VOC2007/ann_img/'
model_url = 'weights/downstream/model_base_retrieval_coco.pth'
model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')

model.eval()
model = model.to(device='cpu')
caption = voc_classes
print('text: %s' % caption)
save_path = 'F:\object-detection\datasets\VOCdevkit\VOC2007/ann_itc_emd/'


for i in range(20):
    f1 = open(save_path+str(i)+'_match.txt', 'w')
    f2 = open(save_path+str(i)+'_mismatch.txt', 'w')
    for img in os.listdir(image_path+str(i)):
        print(img)
        image = load_demo_image(image_size=image_size,device=device,image_path=image_path+str(i)+'/'+img)
        with torch.no_grad():
            image_feat, text_feat, itc_score = model(image,caption,match_head='itc')
            #print(image_feat.shape)
        argmax_itc = itc_score.numpy().astype(np.float32).argmax()
        img_string = ' '.join([str(item) for item in image_feat[0].tolist()])
        if argmax_itc == i:
            f1.write(img_string+'\n')
        else:
            f2.write(img_string+'\n')
    f1.close()
    f2.close()
