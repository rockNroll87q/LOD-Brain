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
	
	In `vol_in` you can pass a folder, for example `vol_in=/data/`, or a volume like `vol_in=/data/sub-01_T1w.nii.gz`





## FAQ

A. If you get the error `ModuleNotFoundError: No module named 'tensorflow_addons'`

~~~
$ singularity shell --cleanenv --nv LOD_Brain.simg
$ python -m pip install tensorflow-addons
~~~


























