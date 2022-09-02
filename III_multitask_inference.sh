mkdir -p $OUTPUT_DIR
# chmod 777 $OUTPUT_DIR

for video in $(ls $TEST_DIR)
do
    if [ -d $TEST_DIR/$video/rgb ]; then
        mkdir -p $OUTPUT_DIR/$video/segmentation
        # chmod 777 $OUTPUT_DIR/$video/segmentation
    fi
done


# sh III_multitask/proposal_inference.sh

sh III_multitask/multitask_recognition_inference.sh