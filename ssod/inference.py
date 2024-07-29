from mmdet.apis import DetInferencer

# Initialize the DetInferencer
#inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')
#models = DetInferencer.list_models('mmdet')

# Perform inference
# inferencer('demo/demo.jpg', out_dir='./output')
# from PIL import Image
# Image.open('./output/vis/demo.jpg')

# checkpoint = 'C:\HQQ\SSOD\MixPL\work_dirs\mixpl_fcos_r50-caffe_fpn_90k_voc-s1-p50/iter_90000.pth'
# config_path = 'C:\HQQ\SSOD\MixPL\projects\MixPL\configs/mixpl_fcos_r50-caffe_fpn_90k_voc-s1-p50.py'
checkpoint = "C:\HQQ\SSOD\MixPL\work_dirs/mixpl_tood_r50-caffe_fpn_180k_coco-s1-p10-both/iter_90000.pth"
config_path = "C:\HQQ\SSOD\MixPL\projects\MixPL\configs\mixpl_tood_r50-caffe_fpn_180k_coco-s1-p10.py"
inferencer = DetInferencer(model=config_path, weights=checkpoint)

inferencer('C:\HQQ\SSOD\MixPL\data\VOCdevkit\VOC2007\yolo_split\images/val/', out_dir='./test_results/mixpl_tood/', no_save_vis=False, no_save_pred=False)
