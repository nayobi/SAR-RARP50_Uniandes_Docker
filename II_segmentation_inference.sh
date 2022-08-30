#/usr/bin/sh

CONFIG_PATH="II_segmentation/S_CONFIG.yaml"
CHECK_POINT="II_segmentation/segmentation_model.pth"
#-------------------------

mkdir -p $OUTPUT_DIR
chmod 777 $OUTPUT_DIR

for video in $(ls $TEST_DIR)
do
    # if [ $video == *"video_"* ]; then
    mkdir -p $OUTPUT_DIR/$video/segmentation
    chmod 777 $OUTPUT_DIR/$video/segmentation
    # fi
done

CUDA_VISIBLE_DEVICES=0 python -W ignore II_segmentation/test_net.py \
--config-file $CONFIG_PATH \
--num-gpus 1 \
DATALOADER.NUM_WORKERS 2 \
SOLVER.IMS_PER_BATCH 10 \
MODEL.WEIGHTS $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
