import cv2
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# 输入是图像和对应的yolo形式的标注文件
# 输出是根据每个标注得到的剪切图像经过resize和对应的类别

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        #print(np.shape(image))
        return image 
    else:
        image = image.convert('RGB')
        return image 

class BLIPDataset(Dataset):
    def __init__(self, img_path, ann_folder, confidence = 0.5, input_shape = (384, 384), train = False):
        self.file_folder        = ann_folder
        self.img_path           = img_path
        self.file_list          = os.listdir(ann_folder)
        self.input_shape        = input_shape
        self.train              = train
        self.length             = len(self.file_list)
        self.confidence         = confidence

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #print(self.file_list[index])
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        name = self.file_list[index]
        image, conf = self.get_crop_img_data(self.file_list[index], self.input_shape)
        return image, conf, name

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def pad_image(self, image, target_size):
        iw, ih = image.size  # 原始图像的尺寸
        w, h = target_size  # 目标图像的尺寸
        scale = min(w / iw, h / ih)  # 转换的最小比例

        # 保证长或宽，至少一个符合目标图像的尺寸
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
        #image.show()
        new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
        # // 为整数除法，计算图像的位置
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
        #new_image.show()

        return new_image

    def load_demo_image(self, device, image, boxes, image_size=(384,384)):
        transform = transforms.Compose([
            transforms.Resize(image_size,interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        length = boxes.shape[0]
        crop_all = np.zeros((length, image_size[0], image_size[1], 3))
        for i, box in enumerate(boxes):
            #print(box)
            crop = image.crop(box)#([box[0],box[1],box[2],box[3]])
            crop = self.pad_image(crop, image_size)
            crop = cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR)
            #crop.save('F:\object-detection\datasets\VOCdevkit\VOC2007\\ann_img/1.jpg')
            #crop = transform(crop)
            #print(crop)
            #crop = np.transpose(np.array(crop, dtype=np.float32), (2, 0, 1))
            crop_all[i] = crop

        return crop_all
        

    def get_crop_img_data(self, file, input_shape):
        image = Image.open(self.img_path+file[:-4]+'.jpg')#.convert('RGB') 
        
        (width, height, _) = np.shape(image)
        # image   = cv2.imread(self.img_path+file[:-4]+'.jpg')
        # print(self.img_path+file[:-4]+'.jpg') 
        # print(image.shape)       
        # 获得预测框，并根据置信度进行筛选
        with open(self.file_folder+file,'r') as f:
            ann = f.readlines()
        ann = [x.split() for x in ann]
        ann_list = []
        class_list = []
        #print(ann)
        for x in ann:
            if float(x[0]) > -1:
                #bbox_xywh = [float(x[1])*width, float(x[2])*height, float(x[3])*width, float(x[4])*height]
                #bbox_xyxy = [bbox_xywh[0]-bbox_xywh[2]/2.0, bbox_xywh[1]-bbox_xywh[3]/2.0, bbox_xywh[0]+bbox_xywh[2]/2.0, bbox_xywh[1]+bbox_xywh[3]/2.0]
                ann_list.append([float(x[1]), float(x[2]), float(x[3]), float(x[4])])
                class_list.append(int(x[0]))    
            else:
                pass
        #print(ann_list)
        # print(ann_list)
        # print(conf_list)
        
        boxes     = np.array([np.array([int(x) for x in box]) for box in ann_list])
        #print(boxes)
        classes   = np.array([x for x in class_list])
        crop_all  = self.load_demo_image('cpu', image, boxes)

        return crop_all, classes

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    crops = []
    confs = []
    names = []
    for crop, conf, name in batch:
        crops.append(crop)
        confs.append(conf)
        names.append(name)
        
    crops = np.array(crops).astype(np.float32)#torch.from_numpy(
    confs = np.array(confs)
    return crops, confs, names

if __name__=='__main__':
    img_path = 'F:\object-detection\datasets\VOCdevkit\VOC2007\JPEGImages/'
    ann_folder = 'F:\object-detection\datasets\VOCdevkit\VOC2007\\ann/'
    save_dir = 'F:\object-detection\datasets\VOCdevkit\VOC2007\\ann_img/'
    data = BLIPDataset(img_path, ann_folder)
    print('dataset done!')
    gen = DataLoader(data, shuffle = False, batch_size = 1, num_workers = 2, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate, 
                                )
    for iteration, batch in enumerate(gen):
        crops, confs, names = batch[0][0], batch[1][0], batch[2][0]
        
        for i in range(len(crops)):
            crop = crops[i]
            #print(type(crop))
            cls = confs[i]
            #print(cls)
            name = names
            #print(name)
            if not os.path.exists(save_dir+str(cls)+'/'):
                os.makedirs(save_dir+str(cls)+'/')
            cv2.imwrite(save_dir+str(cls)+'/'+name[:-4]+'-'+str(i)+'.jpg',crop)
            


