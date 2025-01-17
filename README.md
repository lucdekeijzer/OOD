This repository is meant as a starting point for Out-of-Distribution research.
It contains code to perform inferences, training networks and visualization of results in plots. The code contains two different working detectors, those being the Multi-layer Mahalanobis detector and the Monte-Carlo dropout detector. It is based on the pytorch-ood package in python.

Weights for the network with different classes left out can be found here:
https://drive.google.com/drive/folders/1J0bVwupYEhkbOWiL8JhsTuTZIdWWo8TM?usp=sharing

The current version of this code is in a Jupyter notebook and the different functions are in different code cells. This was done to make sure it can be run on any machine, no matter the GPU, through Google Colab. 

A few things to note:
1. Always make sure the EXCLUDE_CLASS and model weights are the same class, otherwise the detector messess up
2. The current models are trained on a CUDA backend, make sure you have a CUDA environment
3. Check the remapped labels to make sure the classes are continuous (i.e. [0,1,2,3,4,5] instead of [0,1,2,4,5,6]. There is a print statement which will allow you to final check that this happened properly

A final thing which might be helpful, is to put the initialization of the data loaders, detectors and models etc in their own code block so they won't have to be initialized every time you run an inference.
