#/usr/bin/sh

CONFIG_PATH="II_segmentation/S_CONFIG.yaml"
CHECK_POINT="models/segmentation_model.pth"
#-------------------------

mkdir -p $OUTPUT_DIR
# chmod 777 $OUTPUT_DIR

for video in $(ls $TEST_DIR)
do
    if [ -d $TEST_DIR/$video/rgb ]; then
        mkdir -p $OUTPUT_DIR/$video/segmentation
    # chmod 777 $OUTPUT_DIR/$video/segmentation
    fi
done

python -W ignore II_segmentation/test_net.py \
--config-file $CONFIG_PATH \
DATALOADER.NUM_WORKERS $WORKERS \
SOLVER.IMS_PER_BATCH $BATCH \
MODEL.WEIGHTS $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
