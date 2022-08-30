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
import cv2
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
        for video in os.listdir(test_dir):
            for image in os.listdir(os.path.join(test_dir,video,'rgb')):
                frame_num = int(image.split('.')[0])
                if frame_num%60 == 0:
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

def encodeMask(bin_mask):
    """
        Encode a binary mask to RLE for file saving
    """
    rle = m.encode(np.asfortranarray(bin_mask))
    rle['counts'] = str(rle['counts'])
    return rle 


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
    print('Beggining Instruments Segmentation Inference on {} videos ...'.format(os.getenv('TEST_DIR')))
    cfg = setup(args)
    thresh_dict = {k: 0.9 if k<3 else 0.75 for k in range(9)}
    with torch.no_grad():
        model = MaskFormer(cfg)

        print('Loading model from {} ...'.format(cfg.MODEL.WEIGHTS))
        model_state_dict = torch.load(cfg.MODEL.WEIGHTS)
        missing,unespected = model.load_state_dict(model_state_dict)
        print('Missing keys: {}'.format(missing), 'Unescpected keys: {}'.format(unespected))

        if args.num_gpus>0:
            model.cuda()
        dataset = Loader(os.getenv('TEST_DIR'))
        inference_set = torch.utils.data.DataLoader(dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True, collate_fn=collate_funxt)
        model.eval()
        for inputs in tqdm(inference_set,desc='Instrument Segmentation Predictions...'):

            #Model predictions
            output = model(inputs)

            for oid,out in enumerate(output):
                file = inputs[oid]['file_name']
                #Filter mask predictions
                instances = [(out['instances']['scores'][i], out['instances']['pred_classes'][i], mask) 
                            for i,mask in enumerate(out['instances']['pred_masks']) 
                            if out['instances']['scores'][i]>thresh_dict[out['instances']['pred_classes'][i].cpu().item()]]
                
                #Sort predictions for final inference
                instances.sort(key = lambda x: x[0])

                #Saving semantic segmentation image
                sem_im = np.zeros((1080,1920))
                for _,category,mask in instances:
                    mask = mask.cpu().numpy().astype('uint8')
                    category = category.cpu().item()
                    sem_im[mask==1] = category+1
                cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,file),sem_im)
    print('Instrument Segmentation Inference has finished.\n','Segmentation preditions at {}/video_*/segmentation'.format(cfg.OUTPUT_DIR))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    main(args)
