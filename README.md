# Uniandes Team: SAR-RARP50 

### Info
**Memebers**: Nicolás Ayobi, Alejandra Pérez, Juanita Puentes, Santiago Rodríguez, Pablo Arbeláez
**Team Name**: Uniandes 
**Affiliation**: University of Los Andes - Center for Research and Formation in Artificial Intelligence (CinfonIA)
**Contact email**: n.ayobi@uniandes.edu.co
**Challenge**: SAR-RARP50
**Participating Sub-Challenges**: Action Recognition, Instrument Segmentation & Multitask (AR + IS)

## Introduction
Hello!

This is the docker code repository of the Uniandes team for the SAR-RARP50 challenge.

We intend to compete in the three sub-challenges; hence this docker performs inference for action recognition, instrument segmentation, 
and both at the time.

## Running the docker:

First, you need to have the docker image. The image can be taken from https://www.mydrive.com. Alternatively, you can build the docker image with this repository. First, you'll need to download all the model weights at https://www.mydrive.com and place them in a directory named "models" in this repository's main directory. After that, you can build the image by entering the main directory and running
```sh
docker image build –t <tag-name> .
```

After getting the image, we can run the inference codes with docker. The Dockerfile does not have an entry point, so it is necessary to run each inference command separately. However, all tasks can be run with the inference.py script. This script has three arguments: 


1) ```test_dir``` [test_directoy]: (Mandatory) This is the directory with the test data to perform inference on. Please remember that **THIS DOCKER DOES NOT SAMPLE THE 10Hz RGB FRAMES** from the raw videos. Hence, the test directory must have the sampled 10Hz RGB frames inside a directory named "rgb" for each video. That means the test directory must have the following structure:

```tree
test_dir:
|
|--video_x
|        |---rgb
|              |--000000000.png
|              |--000000006.png
|              |--000000012.png
|              ...
|--video_y
|        |---rgb
|              |--000000000.png
|              |--000000006.png
|              |--000000012.png
...            ...
```
2) ```out_dir``` ["output directory"]: (Mandatory) This is the directory where to deposit the inference files for the task. If the directory doesn't exist, then it will be created. For each directory or file in the test directory, the code will create a directory with the same name inside the output directory. Then, the prediction files with the expected format for each video will be put inside its corresponding video path in the output directory. Some additional directories will be created containing the necessary files for the code to run. Make sure to have the proper permissions on the parent directory of the output directory to avoid permission errors.

3) ```--tasks:``` (Optional) This argument specifies which tasks to perform. The possible values tasks are "action_recognition", "segmentation", and "multitask." In this case, the tasks must be input as a single string where all the tasks to perform are separated by a comma (","). For example, inputting "segmentation,multitask" will perform instrument segmentation inference and then multitask inference on the test data. In the same way, "action_recognition" will only perform action recognition inference. All the 15 possible permutations of the tasks are set in the choices of the argument to avoid errors. Please bare in mind that **IF YOU CHOOSE TO PERFORM AN INDIVIDUAL TASK INFERENCE FOLLOWED BY A MULTITASK INFERENCE ON THE SAME OUTPUT DIRECTORY, THEN THE PREDICTION FILES OF THE INDIVIDUAL TASKS WILL BE OVERWRITTEN BY THE PREDICTIONS OF THE MULTITASK. THEREFORE, WE STRONGLY RECOMMEND NOT TO SET THIS ARGUMENT TO ANY COMBINATION OF THE THREE POSSIBLE TASKS.** Instead, we recommend performing any combination of the individual tasks (i.e. --tasks "action_recognition,segmentation") on one output directory and only the multitask inference (--tasks "multitask") on another directory.

Finally, to run the inference script with the docker image, you must run the python inference.py <argumens> on the docker container. This docker creates several new directories and files and some manual CUDA/C++ PyTorch extensions; hence, it is necessary to have the right permissions to perform these tasks. For this reason, we recommend running the cocker image without a user id to have root privileges or running with sudo permissions with a sudo user id. Now, you can run the inference code in the docker container with
```sh
docker container run --rm --gpus=all  \
-v /path/to/data/dir:/data_dir/ \
<tag-name> \
python inference.py data_dir/test_dir data_dir/out_dir \
--tasks <task> [or "<task1>,<task2>"]
```

And that's all!. 
The inference codes are set to perform inference with PyTorch on GPUs. We set the inference to use only one GPU to avoid parallelization or CUDA errors. Similarly, we set a batch size and the number of workers to 2 for all tasks to avoid memory errors; however, this might be pretty slow (2h 30min for the multitask inference according to our tests). However, the batch size and the number of workers can be changed. The docker repository has a .sh file for each possible task; inside, you can find the running settings. You can modify this setting manually and rebuild the docker image before running the code. These .sh files are inside the "III_multitask" directory for multitasking.

# Running the code without docker

The process is very similar if you wish to run the code without the docker. However, you first will have to install all the requirements. The main requirements are:
- Python >= 3.8
- PyTorch >= 1.9
- Torchvision and Cudatoolkit match the PyTorch version
- Numpy 
- OpenCV
- fvcore
- GCC >= 4.9
- tqdm
- psutil
- scikit-image
- cython
- scipy
- shapely
- timm
- h5py
- submitit
- pandas
- python-dateutil
- pytz
- scipy
- six
- tqdm
- typing_extensions

Most of these requirements can be found in the requierements.txt file. Hence, you can set up an environment with all these requirements by running the following.

```sh
conda create --name uniandes_sarrarp python=3.8 -y
conda activate uniandes_sarrarp
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install -U opencv-python
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install -r requirements.txt
```

After installing all the requirements, you need to download all the model weights from http://www.mydrive.com and place them in a new directory named "models" inside the main directory. Now you have to run the infirence.py script. The arguments for this script are explained in the Running the docker section. Also, take an additional note regarding some running parameters found at the end of the last section. 

In case of any questions or inconvenience, feel free to email n.ayobi@uniandes.edu.co.

Thank you very much. 

Best regards,
The Uniandes team