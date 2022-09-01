CONFIG_PATH="III_multitask/multitask_recognition/MT_CONFIG.yaml"
CHECK_POINT="models/multitask_recognition_model.pyth"
#-------------------------


python III_multitask/multitask_recognition/format_videos.py $TEST_DIR $OUTPUT_DIR

python -W ignore III_multitask/multitask_recognition/run_net.py \
--cfg $CONFIG_PATH \
DATA_LOADER.NUM_WORKERS $WORKERS \
TEST.BATCH_SIZE $BATCH \
TEST.CHECKPOINT_FILE_PATH $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
