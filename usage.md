---
layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

[<-- main page](https://rocknroll87q.github.io/LOD-Brain/)

<hr>

You can download the Singularity container here: [link](https://osf.io/hnctg/).

While you can find the docker here: [link](https://hub.docker.com/repository/docker/rocknroll87q/cerebrum3t-multisite).


# Testing tutorial

Steps for singularity (similar to docker):

1. Download the container and the code from Github

	`wget https://osf.io/download/v8g9x/ && mv index.html LOD_Brain.simg`
	
	`git clone https://github.com/rockNroll87q/LOD-Brain`
	
2. Prepare your testing data. As an example, I download a volume from a public repository ([link](https://openneuro.org/datasets/ds002207)):

	`wget https://s3.amazonaws.com/openneuro.org/ds002207/sub-01/anat/sub-01_T1w.nii.gz`
	
	`wget https://s3.amazonaws.com/openneuro.org/ds002207/sub-02/anat/sub-02_T1w.nii.gz`
	
3. Run the container:

	~~~
	singularity exec --cleanenv --nv \
	-B {PATH_TO_YOUR_TESTING_DATA}:/data/ \
	-B ./LOD-Brain/src:/source/ \
	-B ./LOD-Brain/trained_model/:/model/ \
	-B {WHERE_YOU_WANT_TO_SAVE_THE_OUTPUT}:/output/ \
	LOD_Brain.simg python /source/inference.py --vol_in=/data/
	~~~	
	
	Example:
	
	~~~
	singularity exec --cleanenv --nv \
	-B ./:/data/ \
	-B ./LOD-Brain/src:/source/ \
	-B ./LOD-Brain/trained_model/:/model/ \
	-B ./:/output/ \
	LOD_Brain.simg python /source/inference.py --vol_in=/data/
	~~~	
	
	
	Example (docker):
	
	~~~
	docker run -it --rm  \
	-v ~/test_lod_brain/input/T1w_256/:/data/ \
	-v ~/test_lod_brain/LOD-Brain/src/:/source/ \
	-v ~/test_lod_brain/LOD-Brain/trained_model/:/model/ \
	-v ~/test_lod_brain/output/:/output/ \
	rocknroll87q/cerebrum3t-multisite:latest python /source/inference.py --vol_in=/data/
	~~~	

	
	In `vol_in` you can pass a folder, for example `vol_in=/data/`, or a volume like `vol_in=/data/sub-01_T1w.nii.gz`


# Training

To train a network from scratch, use the following example:

1. Build the singularity image using docker hub.

	~~~
	singularity build ./LOD_brain.simg docker://rocknroll87q/cerebrum3t-multisite:latest
	~~~

2. Prepare your csv.

3. Launch the training (below the default values):

	~~~
	singularity shell --cleanenv --nv \
	 -B ~/training_data/:/LOD_Brain/data/ \
	 -B ~/src/:/source/ \
	 -B ~/output/:/output/ \
	LOD_brain.simg bash
	
	export WANDB_DIR='/output/multi_data/'
	cd /source/

	python ./main.py --network.num_levels=2 --network.num_initial_filter=8 --network.num_blocks_per_level=3 --network.conv_block=Plain --network.downsampling_factor=4 --training.loss=per_channel_dice_loss --network.conv_repetition=False --network.bn=GN --data.Filename_csv=dataset_short_training_a+a_LODbrain.csv  --data.Path_in_csv=/LOD_Brain/data/analysis/csv/segmentation/ --network.num_classes=7
	~~~	
Please note that you must set up a wandb account and modify the project name in line 65 of `main.py`.





## FAQ

A. If you get the error `ModuleNotFoundError: No module named 'tensorflow_addons'`

~~~
$ singularity shell --cleanenv --nv LOD_Brain.simg
$ python -m pip install tensorflow-addons
~~~


























