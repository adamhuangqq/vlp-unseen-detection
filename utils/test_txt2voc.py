import os,shutil
import cv2
from lxml.etree import Element, SubElement, tostring

def txt_xml(img_path, img_name, txt_path, img_txt, xml_path, img_xml):
    #读取txt的信息
    clas=[]
    img=cv2.imread(os.path.join(img_path,img_name))
    imh, imw = img.shape[0:2]
    print(img.shape)
    txt_img=os.path.join(txt_path,img_txt)
    with open(txt_img,"r") as f:
        for line in f.readlines():
            #print(line)
            line = line.strip('\n')
            list = line.split(" ")
            if float(list[1]) > 0.3:
                clas.append(list)
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    node_filename = SubElement(node_root, 'filename')
    #图像名称
    node_filename.text = img_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(imw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(imh)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    #print(clas)
    for i in range(len(clas)):
        if clas[i][0] == 'background':
            continue
        bbox_x = int(clas[i][2])
        bbox_y = int(clas[i][3])
        bbox_w = int(clas[i][4])
        bbox_h = int(clas[i][5])
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = (clas[i][0])
        node_pose=SubElement(node_object, 'pose')
        node_pose.text="Unspecified"
        node_truncated=SubElement(node_object, 'truncated')
        node_truncated.text="1"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bbox_x)

        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bbox_y)

        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bbox_w)

        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bbox_h)

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    img_newxml = os.path.join(xml_path, img_xml)
    file_object = open(img_newxml, 'wb')
    file_object.write(xml)
    file_object.close()

if __name__ == "__main__":
    #图像文件夹所在位置
    img_path = 'C:\HQQ\SSOD\RTDETR-main\dataset\images/val/'
    #标注文件夹所在位置
    txt_path = 'C:\HQQ\SSOD\RTDETR-main/runs/val/r50/pred_yolo_pl/'    
    #txt转化成xml格式后存放的文件夹
    xml_path = 'C:\HQQ\SSOD\RTDETR-main/runs/val/r50/pred_voc_pl/'
    for img_name in os.listdir(img_path):
        print(img_name)
        img_xml=img_name[:-4]+".xml"
        img_txt=img_name[:-4]+".txt"
        txt_xml(img_path, img_name, txt_path, img_txt, xml_path, img_xml)
       
