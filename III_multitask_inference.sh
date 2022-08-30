mkdir -p $OUTPUT_DIR
chmod 777 $OUTPUT_DIR

for video in $(ls $TEST_DIR)
do
    # if [ $video == *"video_"* ]; then
    mkdir -p $OUTPUT_DIR/$video/segmentation
    chmod 777 $OUTPUT_DIR/$video/segmentation
    # fi
done


CUDA_VISIBLE_DEVICES=0 sh III_multitask/proposal_inference.sh

CUDA_VISIBLE_DEVIDES=0 sh III_multitask/multitask_recognition_inference.sh