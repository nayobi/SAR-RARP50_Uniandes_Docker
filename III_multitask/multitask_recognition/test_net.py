#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import traceback
import numpy as np
import torch
from tqdm import tqdm

import utils.checkpoint as cu
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from models import build_model
from config.defaults import assert_and_infer_cfg
from utils.misc import launch_job
from utils.parser import load_config, parse_args
from utils.compute_all_metrics import evaluate
from copy import copy
import os
import pycocotools.mask as m
import cv2


logger = logging.get_logger(__name__)

def postprocesses_n_save(preds_dict,cfg):
    print('Beggining noise postprocessing...')

    for video in preds_dict:
        preds_dict[video].sort(key=lambda x: x[3])
        cat_preds = np.array([u[0] for u in preds_dict[video]])
        scores = np.array([u[1] for u in preds_dict[video]])
        frames = ['{:0>9}'.format(u[3]*6) for u in preds_dict[video]]

        scores = np.array(scores)
        #Noise filtering by replacing frame neighbour with higher scores
        for _ in tqdm(range(250),desc='{} postprocessing ...'.format(video)):
            new_scores = copy(scores)
            new_preds = copy(cat_preds)
            for idx in range(1,len(scores)-1):
                ant = max(0,idx-1)
                nex = min(len(cat_preds)-1,idx+1)
                if (cat_preds[ant] != cat_preds[idx]) or (cat_preds[idx] != cat_preds[nex]):
                    if cat_preds[ant] == cat_preds[nex] and np.mean([scores[ant],scores[nex]])>scores[idx]:
                        new_preds[idx] = cat_preds[ant] if ant>0 else cat_preds[nex]
                        new_scores[idx] = scores[ant] if ant>0 else scores[nex]
                    elif cat_preds[ant] != cat_preds[nex] and (scores[ant]>scores[idx] or scores[nex]>scores[idx]):
                        if scores[ant] > scores[nex] :
                            new_preds[idx] = cat_preds[ant]
                            arr = [scores[ant],scores[idx]]
                            new_scores[idx] = float(np.mean(arr))
                        elif scores[ant] < scores[nex]:
                            new_preds[idx] = cat_preds[nex]
                            arr = [scores[nex],scores[idx]]
                            new_scores[idx] = float(np.mean(arr))
            scores = new_scores
            cat_preds = new_preds

        with open(os.path.join(cfg.OUTPUT_DIR,video,'action_discrete.txt'),'w') as f:
            lines = ['{},{}\n'.format(frames[i],cat_preds[i]) for i in range(len(cat_preds))]
            f.writelines(lines)
    print('Postprocessing finished.')

def floatbox(box):
    """
        Transform string key box to list of floats box
    """
    keys = box.split(' ')
    return list(map(float,keys))

def boxiou(bb1,bb2):
    """
        Get the IoU of two bboxes
        bb1: Bbox 1
        bb2: Bbox 2
    """

    #Calculate coordinates of the intersection box
    x1 = max(bb1[0],bb2[0])
    y1 = max(bb1[1],bb2[1])
    x2 = min(bb1[2],bb2[2])
    y2 = min(bb1[3],bb2[3])

    #Return 0 if intersection box is outside any box
    if x2<x1 or y2<y1:
        return 0.0

    #Calculate box IoU
    inter = (x2-x1)*(y2-y1)
    area1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    box_iou = inter/(area1+area2-inter)
    assert box_iou>=0 and box_iou<=1
    return box_iou

def getrealbox(boxes,box):
    """
        Get the real original box for a box with small decimal diferences.
    """
    maxiou = 0
    realbox = ''
    for box2 in boxes:
        iou = boxiou(floatbox(box2),box)
        if iou>maxiou:
            maxiou=iou
            realbox=box2
    
    assert maxiou>0.99 and maxiou<=1, 'Incorrect max IoU {}'.format(maxiou)
    return realbox

def saveSegments(preds,cfg):
    print('Beggining proposals to segmentation processing...')
    try:
        features = torch.load(os.path.join(cfg.OUTPUT_DIR,'stuff_MS','features.pth'))
        feat_dict = {}
        for feat in features:
            feat_dict[feat['file_name']] = feat

        for video in preds:
            for image in preds[video]:
                if image[3]%10==0:
                    im_name = image[2]
                    feat = feat_dict[im_name]
                    sem_im = np.zeros((1080,1920))
                    all_segmets = []
                    for bbox,box_pred in zip(image[0],image[1]):
                        if bbox != [0.0,0.0,0.0,0.0]:
                            category = np.argmax(box_pred)
                            score = box_pred[category]
                            real_box = getrealbox(feat['bboxes'].keys(),bbox)

                            if category<9:
                                #Box values are not exactly the same so the real box must be found with IoU
                                segment = feat['segments'][real_box]
                                all_segmets.append((category+1,segment,score))
                    all_segmets.sort(key=lambda x: x[2])

                    sem_im = np.zeros((1080,1920))
                    for segment in all_segmets:
                        mask = m.decode(segment[1]).astype('uint8')
                        sem_im[mask==1]=segment[0]
                    
                    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,im_name),sem_im)
    except:
        traceback.print_exc()
        breakpoint()
    print('Proposals to segmentation finished.')

@torch.no_grad()
def eval_epoch(val_loader, model, cfg):
    """
    Evaluate the model on the test set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled.
    preds_dict = {}
    box_preds_dict = {}
    model.eval()
    for inputs,image_names,sec,video_name,ori_boxes,features,idxs in tqdm(val_loader,desc='Action recognition and proposal classification predictions ...'):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)

        #Predictions for the "gestures" (action recognition) and "tools" (instrument segmentation) tasks
        all_preds = model(inputs,features,idxs)
        preds = all_preds['gestures'][0].cpu().tolist()
        box_preds = all_preds['tools'][0].cpu().numpy()
        assert len(preds)==len(image_names)==len(sec)==len(video_name)
        
        #Save predicitions for later postprocessing and final file saving
        for nid,name in enumerate(video_name):
            if name not in preds_dict:
                preds_dict[name] = []
                box_preds_dict[name] = []
            category = int(np.argmax(preds[nid]))
            preds_dict[name].append((category, preds[nid][category], image_names[nid], sec[nid], name))
            this_box_preds = box_preds[idxs==nid].tolist()
            this_ori_boxes = ori_boxes[idxs==nid].tolist()
            box_preds_dict[name].append((this_ori_boxes, this_box_preds, image_names[nid], sec[nid], name))


    print('Action recognition and sementation inference have finished. Now performing noise postprocessing and proposal processing...')
    
    postprocesses_n_save(preds_dict,cfg)
    print('Action recognition predictions at {}/video_*/action_discrete.txt'.format(cfg.OUTPUT_DIR))

    saveSegments(box_preds_dict,cfg)
    print('Instrument segmentation predictions at {}/video_*/segmentation'.format(cfg.OUTPUT_DIR))


def test(cfg):
    """
    Perform testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Load checkpoints
    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "val")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    logger.info("Data taken from {}".format(os.getenv('TEST_DIR')))

    # # Perform test on the entire dataset.
    with torch.no_grad():
        eval_epoch(test_loader, model, cfg)

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()