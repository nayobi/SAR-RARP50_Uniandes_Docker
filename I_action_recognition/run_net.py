#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from config.defaults import assert_and_infer_cfg
from utils.misc import launch_job
from utils.parser import load_config, parse_args

from test_net import test
import os

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    cfg.OUTPUT_DIR = os.getenv('OUTPUT_DIR')
    cfg.AVA.FRAME_DIR = os.getenv('TEST_DIR')
    cfg.AVA.FRAME_LIST_DIR = os.path.join(os.getenv('OUTPUT_DIR'),cfg.AVA.FRAME_LIST_DIR)

    # Perform multi-clip testing.
    launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()