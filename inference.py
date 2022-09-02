from subprocess import run
import argparse 
import os
# from sample_videos import main

parser = argparse.ArgumentParser()
parser.add_argument('test_dir', type=str, help='Path of directory with the video directories for test')
parser.add_argument('out_dir', type=str, help='Path of directory where to deposit predictions')
parser.add_argument('--tasks', type=str, default='action_recognition,segmentation', choices=['action_recognition','segmentation','multitask',
                                                                                'action_recognition,segmentation','segmentation,action_recognition','segmentation,multitask','multitask,segmentation','multitask,action_recognition','action_recognition,multitask',
                                                                                'action_recognition,segmentation,multitask','action_recognition,multitask,segmentation','segmentation,multitask,action_recognition','segmentation,action_recognition,multitask','multitask,segmentation,action_recognition','multitask,action_recognition,segmentation'], help='Task to infere')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for dataloading')

args = parser.parse_args()

assert os.path.isdir(args.test_dir), 'The test directory {} does not exist'.format(args.test_dir)

if os.getenv('OUTPUT_DIR') is None or os.getenv('OUTPUT_DIR') == '':
    os.environ['OUTPUT_DIR'] = os.path.join(args.out_dir)

if os.getenv('TEST_DIR') is None or os.getenv('TEST_DIR') == '':
    os.environ['TEST_DIR'] = args.test_dir

if os.getenv('BATCH') is None or os.getenv('BATCH') == '':
    os.environ['BATCH'] = str(args.batch_size)

if os.getenv('WORKERS') is None or os.getenv('WORKERS') == '':
    os.environ['WORKERS'] = str(args.num_workers)

tasks = args.tasks.split(',')
# assert len(tasks)<=3, 'Incorrect number of tasks inputed {}'.format(tasks)
for task in tasks:
    if task == 'action_recognition':
        run(['sh','I_action_recognition_inference.sh'])

    elif task == 'segmentation':
        run(['sh','make.sh'])
        run(['sh','II_segmentation_inference.sh'])

    elif task == 'multitask':
        run(['sh','make.sh'])
        run(['sh','III_multitask_inference.sh'])
    elif task=='':
        print('Empty task !')
    else:
        raise ValueError('Incorrect task {}'.format(args.tasks))
