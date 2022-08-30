CONFIG_PATH="I_action_recognition/AR_CONFIG.yaml"
CHECK_POINT="I_action_recognition/action_recognition_model.pyth"
#-------------------------

mkdir -p $OUTPUT_DIR
chmod 777 $OUTPUT_DIR

for video in $(ls $TEST_DIR)
do
    # if [[ $video == *"video_"* ]]; then
    mkdir -p $OUTPUT_DIR/$video
    chmod 777 $OUTPUT_DIR/$video
    # fi
done

python I_action_recognition/format_videos.py $TEST_DIR $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python -W ignore I_action_recognition/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
DATA_LOADER.NUM_WORKERS 10 \
TEST.BATCH_SIZE 15 \
TEST.CHECKPOINT_FILE_PATH $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
