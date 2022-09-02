# Uniandes Team: SAR-RARP50 

### Info
- **Memebers**: Nicolás Ayobi, Alejandra Pérez, Santiago Rodríguez, Juanita Puentes, Pablo Arbeláez
- **Team Name**: Uniandes 
- **Affiliation**: University of Los Andes - Center for Research and Formation in Artificial Intelligence (CinfonIA)
- **Contact email**: n.ayobi@uniandes.edu.co
- **Challenge**: SAR-RARP50
- **Participating Sub-Challenges**: Action Recognition, Instrument Segmentation & Multitask.

## Introduction
Hello!

This is the docker code repository of the Uniandes team for the SAR-RARP50 challenge.

We intend to compete in the three sub-challenges; hence this docker performs inference for action recognition, instrument segmentation, 
and both at the time.

## Running the docker:

First, you need to have the docker image. The image can be taken from https://drive.google.com/drive/folders/107NVWt-2bTZMj8dO87D2M41RKGHqfA4L?usp=sharing. Alternatively, you can build the docker image with this repository. First, you'll need to download all the model weights at https://drive.google.com/drive/folders/107NVWt-2bTZMj8dO87D2M41RKGHqfA4L?usp=sharing and place them in a directory named "models" in this repository's main directory. After that, you can build the image by entering the main directory and running
```sh
docker image build -t <tag-name> .
```

After getting the image, we can run the inference codes with docker. The Dockerfile does not have an entry point, so it is necessary to run each inference command separately. However, all tasks can be run with the inference.py script. This script has five arguments: 


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
2) ```out_dir``` ["output directory"]: (Mandatory) This is the directory where to deposit the inference files for the task. If the directory doesn't exist, then it will be created. The inference code will create a new directory inside the output path for each directory in the test path that has a child directory named "rgb" (a directory for each video directory in the test path). Then, the prediction files with the expected format for each video will be put inside its corresponding video path in the output directory. Some additional directories will be created containing the necessary files for the code to run. Make sure to have the proper permissions on the parent directory of the output directory to avoid permission errors. Consider that if the test directory and the output directory are the same, there is a chance that the files on the test directory are overwritten (specifically annotation files).

3) ```--tasks``` (Optional) This argument specifies which tasks to perform. The possible values tasks are "action_recognition", "segmentation", and "multitask." In this case, the tasks must be input as a single string where all the tasks to perform are separated by a comma (","). For example, inputting ```--tasks segmentation,multitask``` will perform instrument segmentation inference and then multitask inference on the test data. In the same way, ```--tasks action_recognition``` will only perform action recognition inference. All the 15 possible permutations of the tasks are set in the choices of the argument to avoid errors. Please bare in mind that **IF YOU CHOOSE TO PERFORM AN INDIVIDUAL TASK INFERENCE FOLLOWED BY A MULTITASK INFERENCE ON THE SAME OUTPUT DIRECTORY, THEN THE PREDICTION FILES OF THE INDIVIDUAL TASKS WILL BE OVERWRITTEN BY THE PREDICTIONS OF THE MULTITASK. THEREFORE, WE STRONGLY RECOMMEND NOT TO SET THIS ARGUMENT TO ANY COMBINATION OF THE THREE POSSIBLE VALUES.** Instead, we recommend performing the individual tasks (i.e. ```--tasks "action_recognition,segmentation"```) on one output directory and then only the multitask inference (```--tasks multitask```) on another directory.

4) ```--batch_size``` (Optional) This argument specifies the batch size for inference. The default value is 2, enough to fit in a Titan X Pascal GPU (12GB). You can modify this parameter according to your GPU resources. The inference speed may vary with this argument. The following table specifies the GPU expense in MiB of different batch size values. Also, if you specify more than one task in the ```--tasks``` argument or select multitask inference, then the same batch size will be used for all the tasks.

| Batch Size | Segmentation (MiB) | Action Recognition (MiB) |
| ------ | ------ | ----- |
| 2 | 10036 | 3270 |
| 5 | 17140 | 6412 |
| 10 | 28982 | 11648 |
| 15 | 40820 | 16885 |
| 20 | 52659 | 22122 |
| 30 | 76339 | 32834 |

5) ```--num_workers``` (Optional) This argument defines the number of workers for data loading. This argument is of special importance for the action recognition tasks as it significantly speeds inference. The default value is 10. Hence it is necessary to increase the shared memory allowance of your container; we recommend setting ```--shm-size=3G``` when running the container. If you further increase the number of workers, you might also have to increase your docker's shared memory allowance. 

Finally, to run the inference script with the docker image, you must run ```python inference.py <argumens>``` on the docker container. This docker creates several new directories and files and some manual CUDA/C++ PyTorch extensions; hence, it is necessary to have the right permissions to perform these actions. For this reason, we recommend running the cocker image without a user id to have root privileges or running with sudo permissions with a sudo user id. Now, you can run the inference code in the docker container with
```sh
docker container run --rm --gpus=all --shm-size=3G \
-v /path/to/data/dir:/data_dir/ \
<tag-name> \
python inference.py /data_dir/test_dir /data_dir/out_dir \
--tasks <task> [or "<task1>,<task2>"]
```

And that's all!. 
The inference codes are set to perform inference with PyTorch on GPUs. We set the inference to use only one GPU to avoid parallelization or CUDA errors. As stated previously, the batch size and the number of workers can be modified with the input parameters; make sure to use the correct values to avoid out-of-memory errors. 

# Running the code without docker

The process is similar if you wish to run the code without the docker. However, you first will have to install all the requirements. The main requirements are:
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

After installing all the requirements, you need to download all the model weights from https://drive.google.com/drive/folders/107NVWt-2bTZMj8dO87D2M41RKGHqfA4L?usp=sharing and place them in a new directory named "models" inside the main directory. Now you have to run the infirence.py script. The arguments for this script are explained in the Running the docker section. Also, take an additional note regarding some running parameters found at the end of the last section. 

In case of any questions or inconvenience, feel free to email n.ayobi@uniandes.edu.co.

Thank you very much. 

Best regards,
The Uniandes team