import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/r18/ori-pretrained-bifpn/weights/last.pt')
    model.val(data='dataset/data_od.yaml',
              split='test',
              imgsz=640,
              batch=4,
              save_json=False, # if you need to cal coco metrice
              project='runs/val/r18',
              name='ori-pretrained-bifpn',
              )