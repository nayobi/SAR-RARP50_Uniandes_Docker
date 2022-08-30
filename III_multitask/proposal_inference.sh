#/usr/bin/sh

CONFIG_PATH="III_multitask/region_proposals/S_CONFIG.yaml"
CHECK_POINT="III_multitask/region_proposals/segmentation_model.pth"
#-------------------------

python -W ignore III_multitask/region_proposals/test_net.py \
--config-file $CONFIG_PATH \
--num-gpus 1 \
DATALOADER.NUM_WORKERS 5 \
SOLVER.IMS_PER_BATCH 10 \
MODEL.WEIGHTS $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
