# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import torch
import os
import skimage.io as io
from tqdm import tqdm
import pycocotools.mask as m
import numpy as np

from defaults import get_cfg
from utils.detectron2_utils import default_argument_parser
from config import add_maskformer2_config,add_deeplab_config
from maskformer_model import MaskFormer
import json
import pickle as pkl
import cv2
import csv
from PIL import Image

class Loader(torch.utils.data.Dataset):
    """
    Dataset class for video frames loading
    """
    def __init__(self,test_dir):
        super().__init__()

        self.test_dir = test_dir
        self.getPaths(test_dir)
        print(test_dir)

    def getPaths(self,test_dir):
        self.im_list = []
        # self.all_ims_list = []
        test_list = os.listdir(test_dir)
        test_list.sort()
        for video in test_list:
            if os.path.isdir(os.path.join(test_dir,video,'rgb')):
                frames_list = os.listdir(os.path.join(test_dir,video,'rgb'))
                frames_list.sort()
                for image in frames_list:
                    frame_num = int(image.split('.')[0])
                    if frame_num%6 == 0:
                        self.im_list.append(('{}/rgb/{}'.format(video,image),video,frame_num))
    
    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx):
        im_path,video,frame = self.im_list[idx]
        image = io.imread(os.path.join(self.test_dir,im_path))
        H,W,_ = image.shape
        image = np.asarray(Image.fromarray(image).resize((1333,750)))
        image = torch.tensor(image).permute((2,0,1))
        return {'image':image, 'width': W, 'height': H, 'file_name': '{}/segmentation/{:0>9}.png'.format(video,frame), 'file_path': im_path, 'video': video, 'frame':frame}

def collate_funxt(batch):
    return batch

def encodeMask(bin_mask,w,h):
    """
        Encode a binary mask to RLE for file saving and return bbox
    """
    rle = m.encode(np.asfortranarray(bin_mask))
    ys,xs = np.where(bin_mask==1)
    x1, y1, x2, y2 = map(int,(np.min(xs),np.min(ys),np.max(xs),np.max(ys)))
    x1 /= w
    x2 /= w
    y1 /= h
    y2 /= h

    bbox = [x1,y1,x2,y2]
    return rle, bbox


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    add_maskformer2_config(cfg)
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg


def main(args):
    print('Beggining Instruments Region Proposals Inference on {} videos ...'.format(os.getenv('TEST_DIR')))
    cfg = setup(args)
    features_list = []
    csv_boxes = []
    with torch.no_grad():
        model = MaskFormer(cfg)
        print('Loading model from {} ...'.format(cfg.MODEL.WEIGHTS))
        model_state_dict = torch.load(cfg.MODEL.WEIGHTS)
        missing,unespected = model.load_state_dict(model_state_dict)
        print('Missing keys {}'.format(missing),'Unespected keys {}'.format(unespected))

        if args.num_gpus>0:
            model.cuda()
        dataset = Loader(os.getenv('TEST_DIR'))
        inference_set = torch.utils.data.DataLoader(dataset, batch_size=int(cfg.SOLVER.IMS_PER_BATCH), num_workers=int(cfg.DATALOADER.NUM_WORKERS), pin_memory=True, collate_fn=collate_funxt)
        model.eval()
        box_count = 0
        for inputs in tqdm(inference_set,desc='Instrument regions predictions...'):
            output = model(inputs)
            # breakpoint()
            for oid,out in enumerate(output):
                file = inputs[oid]['file_name']
                this_dict = {'file_name': file, 'video': inputs[oid]['video'], 'frame': inputs[oid]['frame'],
                            'height': inputs[oid]['height'], 'width': inputs[oid]['width'], 
                            'bboxes': {}, 'segments': {}, 'score':{}, 'category':{}}

                instances = [(out['instances']['scores'][i].cpu().item(), 
                              out['instances']['pred_classes'][i].cpu().item(), 
                              out['instances']['mask_embed'][i].cpu().tolist(), 
                              mask.cpu().numpy().astype('uint8')) 
                              for i,mask in enumerate(out['instances']['pred_masks']) 
                              if out['instances']['scores'][i]>=0.75]

                for score,category,feature,mask in instances:
                    box_count += 1
                    mask,bbox = encodeMask(mask, inputs[oid]['width'], inputs[oid]['height'])
                    bbox_key = '{} {} {} {}'.format(*bbox)
                    this_dict['bboxes'][bbox_key] = feature
                    this_dict['segments'][bbox_key] = mask
                    this_dict['score'][bbox_key] = score
                    this_dict['category'][bbox_key] = category
                    csv_boxes.append((inputs[oid]['video'], int(inputs[oid]['frame']/6), box_count, bbox[0], bbox[1], bbox[2], bbox[3]))
                
                features_list.append(this_dict)
                if len(instances)==0:
                    box_count +=1
                    csv_boxes.append((inputs[oid]['video'], int(inputs[oid]['frame']/6), box_count, 0.0, 0.0, 0.0, 0.0))
                
        os.makedirs(os.path.join(cfg.OUTPUT_DIR,'stuff_MS'),exist_ok=True)
        torch.save(features_list,os.path.join(cfg.OUTPUT_DIR,'stuff_MS','features.pth'))
        with open(os.path.join(cfg.OUTPUT_DIR,'stuff_MS','inference_preds.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerows(csv_boxes)

    print('Instrument Region Proposals Inference has finished.\n','Region proposals at {}/stuff_MS/features.pth'.format(cfg.OUTPUT_DIR))
    return features_list,csv_boxes


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # out_dir = 
    main(args)
    # try:
    #     json.dump(out_dir,open('preds.json','w'))
    # except:
    #     pkl.dump(out_dir,open('preds.pkl','wb'))
