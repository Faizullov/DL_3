import torch.nn.functional as F 
import numpy               as np

import argparse
import torch
import time
import glob
import cv2
import os

from prior_boxes import detect_objects, prior_boxes
from voc_dataset import default_transform 

from model_resnet18 import SSD_ResNet18
from model_vgg16    import SSD_VGG16
from voc_dataset    import VOC_CLASSES

from torchvision.utils import draw_bounding_boxes
from logger            import Logger
from tqdm              import tqdm


parser = argparse.ArgumentParser()

args = parser.parse_args()

dataset_root_dir = 'dataset/JPEGImages' 
output_dir   = 'evaluate'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__': 
    print('using {} to eval, use cpu may take an hour to complete !!'.format(device))
    
    custom_config = {
     'num_classes'  : 3,
     'feature_maps' : [(45,80), (23,40), (12,20), (6,10), (3,5), (2,3)], #ResNet18
     'min_sizes'    : [0.10, 0.20, 0.37, 0.54, 0.71, 1.00],
     'max_sizes'    : [0.20, 0.37, 0.54, 0.71, 1.00, 1.05],
     
     'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
     'num_priors'   : [6, 6, 6, 6, 4, 4],
     'variance'     : [0.1, 0.2],
     'clip'         :    True,
    
     'overlap_threshold': 0.5,
     'conf_threshold'   : 0.1,
     'neg_pos_ratio'    :   3,

     'model_name' : 'resnet18'
    }

    prior_box_s = prior_boxes(custom_config)
    prior_box_s_gpu = prior_box_s.cuda()
    
    model      = SSD_ResNet18(custom_config['num_priors'], custom_config['num_classes'])
   
    checkpoint = f"output/{custom_config['model_name']}.pth"
    print('loading checkpoint from {}'.format(checkpoint))
    
    state_dict = torch.load(checkpoint, map_location=None if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    
    print('model loaded')
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(output_dir):
        os.remove( os.path.join( output_dir, file) )

    print('start detect.........')
    
    for i, image_filename in tqdm(enumerate( glob.glob("dataset/JPEGImages/*.jpg") )):
        source_image_numpy = cv2.imread(image_filename)
        
        source_image_numpy = source_image_numpy.astype(np.float32) / 255.
        source_image_cpu   = default_transform(source_image_numpy)
        source_image_gpu   = source_image_cpu.to(device)
        
        source_height, source_width = source_image_cpu.shape[1], source_image_cpu.shape[2]
        
        with torch.no_grad():
            loc, conf = model(source_image_gpu.unsqueeze(0))
            
            conf = F.softmax(conf, dim=2).cpu()
            loc  = loc                   .cpu()
        
        target_box_ss, target_label_ss, target_conf_ss = detect_objects(loc, conf, prior_box_s, 3, 0.5, 0.1)
        target_box_s, target_label_s, target_conf_s = target_box_ss[0], target_label_ss[0], target_conf_ss[0]
        
        if torch.numel(target_box_s)>0 and torch.numel(target_label_s)>0 and torch.numel(target_conf_s)>0 :
            target_class_s = [VOC_CLASSES[label] for label in target_label_s.tolist() ]
            
            target_image_cpu   = torch.from_numpy(source_image_numpy * 255).to(torch.uint8).permute([2,0,1])
            target_box_s       = target_box_s * torch.as_tensor([[source_width, source_height, source_width, source_height]])
            
            target_image_cpu   = draw_bounding_boxes(target_image_cpu, target_box_s, labels=target_class_s, colors='red', width=3)
            target_image_numpy = target_image_cpu.permute([1,2,0]).numpy()
        else:
            target_image_numpy = target_image_cpu.permute([1,2,0]).numpy()
        
        cv2.imwrite( os.path.join(output_dir, f'result_{i}.png'), target_image_numpy) 

