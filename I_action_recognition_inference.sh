CONFIG_PATH="I_action_recognition/AR_CONFIG.yaml"
CHECK_POINT="models/action_recognition_model.pyth"
#-------------------------

mkdir -p $OUTPUT_DIR
# chmod 777 $OUTPUT_DIR

for video in $(ls $TEST_DIR)
do
    if [ -d $TEST_DIR/$video/rgb ]; then
        mkdir -p $OUTPUT_DIR/$video
    # chmod 777 $OUTPUT_DIR/$video
    fi
done

python I_action_recognition/format_videos.py $TEST_DIR $OUTPUT_DIR

python -W ignore I_action_recognition/run_net.py \
--cfg $CONFIG_PATH \
DATA_LOADER.NUM_WORKERS $WORKERS \
TEST.BATCH_SIZE $BATCH \
TEST.CHECKPOINT_FILE_PATH $CHECK_POINT \
OUTPUT_DIR $OUTPUT_DIR
