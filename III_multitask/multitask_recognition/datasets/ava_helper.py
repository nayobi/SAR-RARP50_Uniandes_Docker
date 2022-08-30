#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from cv2 import BOWKMeansTrainer
import torch
import logging
from collections import defaultdict

from utils.env import pathmgr

logger = logging.getLogger(__name__)

def load_features_boxes(cfg):
    """
    Load boxes features from faster cnn trained model. Used for initialization
    in actions and tools detection.

    Args:
        cfg (CfgNode): config.

    Returns:
        features (tensor): a tensor of faster weights.
    """
    features = torch.load(cfg.FASTER.FEATURES_VAL)
    real_features = {}
    for feat in features:
        file_name = feat['file_name']
        assert file_name not in real_features, '{}'.format(file_name)
        real_features[file_name]=feat['bboxes']


    return real_features

def load_image_lists(cfg):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    # breakpoint()
    list_filenames = [
        os.path.join(cfg.AVA.FRAME_LIST_DIR, filename)
        for filename in cfg.AVA.TEST_LISTS
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        with pathmgr.open(list_filename, "r") as f:
            # f.readline()
            for line in f:
                row = line.split()
                assert len(row) == 4
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]
                image_paths[data_key].append(row[3])


    image_paths = [image_paths[i] for i in range(len(image_paths))]
    logger.info("Finished loading image paths from: %s" % ", ".join(list_filenames))

    return image_paths, video_idx_to_name

def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    # gt_lists = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == "train" else []
    pred_lists = (
        cfg.AVA.TEST_PREDICT_BOX_LISTS
    )
    # breakpoint()
    ann_filename = os.path.join(cfg.AVA.ANNOTATION_DIR, pred_lists[0])

    # Only select frame_sec % 4 = 0 samples for validation if not

    all_boxes, count, unique_box_count = parse_bboxes_file(
        ann_filename=ann_filename
    )
    logger.info("Finished loading annotations from: %s" % ann_filename)
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """

        # PSI-AVA contains one frame per second
        return int(sec)

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        # sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx]:
            sec_idx = list(boxes_and_labels[video_idx]).index(sec)
            if len(boxes_and_labels[video_idx][sec])>0:
                keyframe_indices.append((video_idx, sec_idx, sec, sec_to_frame(sec)))
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels

def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        try:
            count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
        except:
            breakpoint()
    return count

def parse_bboxes_file(ann_filename):
    """
    Parse PSI-AVA bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of PSI-AVA bounding boxes annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """
    all_boxes = {}
    count = 0
    unique_box_count = 0

    with pathmgr.open(ann_filename, "r") as f:
        for line in f:
            row = line.strip().split(",")
            assert len(row)==7
            # When we use predicted boxes to train/eval, we need to
            # save as empty the boxes whose scores are below the threshold.
            # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
            box_key = ",".join(row[3:7])
            box = list(map(float, row[3:7]))

            video_name, frame_sec = row[0], int(row[1])

            if video_name not in all_boxes:
                all_boxes[video_name] = {}
            if frame_sec not in all_boxes[video_name]:
                all_boxes[video_name][frame_sec] = {}
                # for sec in range(int(len(os.listdir('/media/SSD0/nayobi/All_datasets/SAR-RARP50/videos/fold1/{}/segmentation'.format(video_name))))):
                #     all_boxes[video_name][sec*10] = {}

            if box_key not in all_boxes[video_name][frame_sec]:
                all_boxes[video_name][frame_sec][box_key] = box
                unique_box_count += 1

            count += 1

    new_all_boxes = {}
    for video_name in all_boxes:
        new_all_boxes[video_name]={}
        for frame_sec in all_boxes[video_name]:
            if len(all_boxes[video_name][frame_sec])>0:
                new_all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    return new_all_boxes, count, unique_box_count