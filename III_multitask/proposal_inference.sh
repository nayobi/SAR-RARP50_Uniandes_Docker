#/usr/bin/sh

CONFIG_PATH="III_multitask/region_proposals/S_CONFIG.yaml"
CHECK_POINT="models/region_proposal_model.pth"
#-------------------------

python -W ignore III_multitask/region_proposals/test_net.py \
--config-file $CONFIG_PATH \
DATALOADER.NUM_WORKERS $WORKERS \
SOLVER.IMS_PER_BATCH $BATCH \
MODEL.WEIGHTS $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
