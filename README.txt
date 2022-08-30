Memebers: Nicolás Ayobi, Alejandra Pérez, Juanita Puentes, Santiago Rodríguez, Pablo Arbeláez
Team: Uniandes (University of Los Andes - Center for Research and Formation in Artificial Intelligence (CinfonIA))
Challenge: SAR-RARP50
Participating Sub-Challenges: Action Recognition, Instrument Segmentation & Multitask (AR + IS)



INTRODUCTION:
Hello!

This is the docker repository of the Uniandes team for the SAR-RARP50 challenge.

We intend to compete in the three sub-challenges; hence this docker performs inference for action recognition, instrument segmentation, 
and both at the time.

RUNNING THE CODE:

First, the docker image must be built with:

docker image build –t <tag-name> .

The Dockerfile does not have an entry point, so it is necessary to run each inference command separately. However, all tasks can be run 
with the inference.py script. This script has three arguments: 

1) test_dir: ["test directory"] (mandatory) This is the directory with the test data to perform inference on. Bare in mind that THIS DOCKER DOES NOT 
             SAMPLE THE 10Hz RGB FRAMES from the raw videos. Hence, the test directory must have the sampled 10Hz RGB frames inside a directory named 
             "rgb" for each video. That means the test directory must have the following structure:

 	  test_dir:
	  |
          |--video_x
          |        |
          |        |---rgb
          |              |
          |              |--000000000.png
          |              |--000000006.png
          |              |--000000012.png
          |              ...
          |
          |--video_y
          |        |
          |        |---rgb
          |              |
          |              |--000000000.png
          |              |--000000006.png
          |              |--000000012.png
          ...            ...

2) out_dir: ["output directory"] (mandatory) This is the directory where to deposit the inference files for the task. If the directory doesn't exist, 
            then it will be created. For each directory or file in the test directory, the code will create a directory with the same name inside the 
            output directory. Then, the prediction files with the challenge expected format for each video will be put inside its corresponding video 
            path in the output directory. Some additional directories will be created containing the necessary files for the code to run. Make sure to 
            have the proper permissions on the parent directory of the output directory to avoid permission errors. 

3) --tasks: (optional) This argument specifies which tasks to perform. The possible values tasks are "action_recognition", "segmentation", and 
            "multitask." In this case, the tasks must be input as a single string where all the tasks to perform are separated by a comma (","). For 
            example, inputting "segmentation,multitask" will perform instrument segmentation inference and then multitask inference on the test 
            data. In the same way, "action_recognition" will only perform action recognition inference. All the 15 possible permutations of the tasks 
            are set in the choices of the argument to avoid errors. Bare in mind that IF YOU CHOOSE TO PERFORM AN INDIVIDUAL TASK INFERENCE FOLLOWED BY
            A MULTITASK INFERENCE ON THE SAME OUTPUT DIRECTORY, THEN THE PREDICTION FILES OF THE INDIVIDUAL TASKS WILL BE OVERWRITTEN BY THE 
            PREDICTIONS OF THE MULTITASK. THEREFORE, WE STRONGLY RECOMMEND NOT TO SET THIS ARGUMENT TO ANY COMBINATION OF THE THREE POSSIBLE TASKS. 
            Instead, we recommend performing any combination of the individual tasks (e.g. --tasks "action_recognition,segmentation") on one output 
            directory and only the multitask inference (--tasks "multitask") on another directory.

To run the inference script with the docker image, you just have to run the python inference.py <argumens> on the docker container. This is done
with:

docker container run --rm --gpus=all --user=1000:1000 -v /path/to/data/dir:/data_dir/ <tag-name> \
python inference.py data_dir/test data_dir/predictions "<task1>,<task2>"

And that's all!. The inference codes are set to perform inference with PyTorch on GPUs. We set the inference only to use one GPU to avoid 
parallelization or CUDA errors. Similarly, we set a batch size of 2 and only two workers for all tasks to avoid memory errors; however, this might be 
quite slow (2h 30min for the multitask inference according to our tests). However, the batch size and the number of workers configurations can be 
changed. The docker repository has a .sh for each possible task, and inside, you can find the running settings. You can modify this settings manully 
before bulding the docker image or you can change them after and the re build the image. For the multitask, this files are inside the III_multitask
directory.

Thank you very much for you collaboration. In case of any issues or questions just send an email to n.ayobi@uniandes.edu.co.

Best regards,
the Uniandes team.

