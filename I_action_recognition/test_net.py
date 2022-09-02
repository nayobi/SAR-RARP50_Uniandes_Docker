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
from copy import copy
import os


logger = logging.get_logger(__name__)

def postprocesses_n_save(preds_dict,cfg):
    print('Beggining noise postprocessing...')
    funct = lambda x: 2*(x[0]*x[1])/(x[1]+x[0])

    try:
        for video in preds_dict:
            preds_dict[video].sort(key=lambda x: x[3])
            cat_preds = np.array([u[0] for u in preds_dict[video]])
            scores = np.array([u[1] for u in preds_dict[video]])
            frames = ['{:0>9}'.format(u[3]*6) for u in preds_dict[video]]

            scores = np.array(scores)

            #Noise filtering by replacing frame neighbour with higher scores
            for _ in tqdm(range(250),desc='{} noise postprocessing ...'.format(video)):
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
                                new_scores[idx] = float(funct(arr))
                            elif scores[ant] < scores[nex]:
                                new_preds[idx] = cat_preds[nex]
                                arr = [scores[nex],scores[idx]]
                                new_scores[idx] = float(funct(arr))
                scores = new_scores
                cat_preds = new_preds

            with open(os.path.join(cfg.OUTPUT_DIR,video,'action_discrete.txt'),'w') as f:
                lines = ['{},{}\n'.format(frames[i],cat_preds[i]) for i in range(len(cat_preds))]
                f.writelines(lines)
    except:
        traceback.print_exc()
        breakpoint()
    print('Postprocessing finished.')
        

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
    model.eval()
    for inputs,image_names,sec,video_name in tqdm(val_loader,desc='Action recognition predictions ...'):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

        #Predictions for the "gestures" (action recognition) task
        preds = model(inputs)['gestures'][0].cpu().tolist()
        assert len(preds)==len(image_names)==len(sec)==len(video_name)

        #Save predicitions for later postprocessing and final file saving
        for nid,name in enumerate(video_name):
            if name not in preds_dict:
                preds_dict[name] = []
            category = int(np.argmax(preds[nid]))
            preds_dict[name].append((category,preds[nid][category],image_names[nid],sec[nid],name))

    #Postprocess complete video predictions to remove noise and over segmentation
    print('Action recognition inference has finished. Now performing noise postprocessing...')
    postprocesses_n_save(preds_dict,cfg)
    print('Action recognition predictions at {}/video_*/action_discrete.txt'.format(cfg.OUTPUT_DIR))




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

    # Load checkpoint
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