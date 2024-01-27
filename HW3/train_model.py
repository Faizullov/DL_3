from torch.optim.lr_scheduler import MultiStepLR
from torchviz                 import make_dot

import numpy as np

import argparse
import torch
import time
import os

from voc_dataloader      import get_train_dataloader, get_test_dataloader

from model_resnet18  import SSD_ResNet18
from model_vgg16     import SSD_VGG16
from multibox_loss   import MultiBoxLoss
from prior_boxes     import prior_boxes
#from logger          import Logger

from tqdm import tqdm

def make_parser():
    parser = argparse.ArgumentParser(description="Train Single Shot MultiBox Detector on custom dataset")
    parser.add_argument('--dataset_root_dir'     , type=str  , default='dataset'                )
    parser.add_argument('--epochs'               , type=int  , default=100                      )
    parser.add_argument('--batch_size'           , type=int  , default=4                        )
    parser.add_argument('--checkpoint'           , type=str  , default=None                     )
    parser.add_argument('--output'               , type=str  , default='output'                 )
    parser.add_argument('--multistep' , nargs='*', type=int  , default=[20, 40, 60            ] )
    parser.add_argument('--learning_rate'        , type=float, default=1e-3                     )
    parser.add_argument('--momentum'             , type=float, default=0.9                      )
    parser.add_argument('--weight-decay'         , type=float, default=0.0005                   )
    parser.add_argument('--warmup'               , type=int  , default=None                     )
    parser.add_argument('--num-workers'          , type=int  , default=4                        )
    
    parser.add_argument('--seed'                 , type=int  , default=42                       )
    
    return parser

def train_process(args):
    torch.manual_seed(     args.seed)
    np.random.seed   (seed=args.seed)   
    
    dataset_root_dir          = args.dataset_root_dir
    train_annotation_filename = os.path.join( dataset_root_dir, "ImageSets/Main/trainval.txt" )
    test_annotation_filename  = os.path.join( dataset_root_dir, "ImageSets/Main/test.txt"     )
    train_dataloader          = get_train_dataloader(dataset_root_dir, train_annotation_filename, args.batch_size, args.num_workers)
    test_dataloader           = get_test_dataloader (dataset_root_dir, test_annotation_filename , args.batch_size, args.num_workers)
    
    learning_rate = args.learning_rate
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    '''    
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
     'neg_pos_ratio'    :   3,
     'model_name' : 'resnet18'
    }
     
    model = SSD_ResNet18(custom_config['num_priors'], custom_config['num_classes'])
    '''
    custom_config = {
     'num_classes'  : 3,
     #'feature_maps' : [(90,160), (45,80), (23,40), (12,20), (10,18), (8,16)], #VGG16 - 1280x720 
     'feature_maps' : [(45,80), (22,40), (11,20), (6,10), (4,8), (2,6)], #VGG16 - 640x360

     'min_sizes'    : [0.10, 0.20, 0.37, 0.54, 0.71, 1.00],
     'max_sizes'    : [0.20, 0.37, 0.54, 0.71, 1.00, 1.05],
     
     'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
     'num_priors'   : [6, 6, 6, 6, 4, 4],
     'variance'     : [0.1, 0.2],
     'clip'         :    True,
    
     'overlap_threshold': 0.5, 
     'neg_pos_ratio'    :   3,

     'model_name' : 'vgg16'
    }
    model = SSD_VGG16(custom_config['num_priors'], custom_config['num_classes'])
    
    prior_box_s = prior_boxes(custom_config)
    prior_box_s_gpu = prior_box_s.cuda()
    
    overlap_threshold = custom_config['overlap_threshold']
    neg_pos_ratio     = custom_config['neg_pos_ratio'    ]
    variance          = custom_config['variance']

    criterion = MultiBoxLoss( overlap_threshold, neg_pos_ratio, variance)
    model    .cuda()
    criterion.cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay )
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.2)
    
    best_loc_loss, best_cls_loss, best_loss = np.inf, np.inf, np.inf
    for epoch in tqdm(list(range(args.epochs))):
        #Train model
        train_loc_loss, train_cls_loss, train_loss = 0, 0, 0
        model.train()
        for i , (image_s_cpu, box_ss_cpu, label_ss_cpu) in enumerate(train_dataloader):
            image_s_gpu  = image_s_cpu .cuda()
            label_ss_gpu = [ label_s_cpu.cuda() for label_s_cpu in label_ss_cpu ]
            box_ss_gpu   = [ box_s_cpu  .cuda() for box_s_cpu   in box_ss_cpu   ]
            
            pred_loc_ss_gpu, pred_conf_ss_gpu = model(image_s_gpu)
            
            loc_loss, cls_loss = criterion( (pred_loc_ss_gpu, pred_conf_ss_gpu, prior_box_s_gpu), (label_ss_gpu, box_ss_gpu))
            loss = loc_loss + cls_loss
            
            optimizer.zero_grad()
            loss     .backward()
            optimizer.step()
            
            train_loc_loss += loc_loss.item()
            train_cls_loss += cls_loss.item()
            train_loss     += loss    .item()
        scheduler.step()
        #Eval model
        eval_loc_loss, eval_cls_loss, eval_loss = 0, 0, 0
        model.eval()
        for i , (image_s_cpu, box_ss_cpu, label_ss_cpu) in enumerate(test_dataloader):
            image_s_gpu = image_s_cpu.cuda()
            label_ss_gpu = [ label_s_cpu.cuda() for label_s_cpu in label_ss_cpu ]
            box_ss_gpu   = [ box_s_cpu  .cuda() for box_s_cpu   in box_ss_cpu   ]
            
            pred_loc_ss_gpu, pred_conf_ss_gpu = model(image_s_gpu)
            
            loc_loss, cls_loss = criterion( (pred_loc_ss_gpu, pred_conf_ss_gpu, prior_box_s_gpu), (label_ss_gpu, box_ss_gpu))
            loss = loc_loss + cls_loss
            
            eval_loc_loss += loc_loss.item()
            eval_cls_loss += cls_loss.item()
            eval_loss     += loss    .item()
        print('epoch[{}] | lr {:.5f} | loc_loss [{:.2f}/{:.2f}] | cls_loss [{:.2f}/{:.2f}] | total_loss [{:.2f}/{:.2f}]'.format(epoch, scheduler.get_last_lr()[0], train_loc_loss, eval_loc_loss, train_cls_loss, eval_cls_loss, train_loss, eval_loss))
        if eval_loss < best_loss :
            torch.save(model.state_dict(), os.path.join(args.output, f"{custom_config['model_name']}.pth"))
            best_loc_loss, best_cls_loss, best_loss = eval_loc_loss, eval_cls_loss, eval_loss

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    
    os.makedirs('./models', exist_ok=True)

    log_interval = 20
    json_summary = 'output.json'
    
    train_process(args)
