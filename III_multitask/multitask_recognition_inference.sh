CONFIG_PATH="III_multitask/multitask_recognition/MT_CONFIG.yaml"
CHECK_POINT="III_multitask/multitask_recognition/multitask_recognition_model.pyth"
#-------------------------


python III_multitask/multitask_recognition/format_videos.py $TEST_DIR $OUTPUT_DIR

python -W ignore III_multitask/multitask_recognition/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
DATA_LOADER.NUM_WORKERS 5 \
TEST.BATCH_SIZE 9 \
TEST.CHECKPOINT_FILE_PATH $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
