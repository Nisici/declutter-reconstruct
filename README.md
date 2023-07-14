# Robust Structure Identification and Room Segmentation of Cluttered Indoor Environments from Occupancy Grid Maps 	

## Introduction

ROSE^2 is a method for RObust StructurE identification and ROom SEgmentation of 2D occupancy maps, which may be cluttered and incomplete. Our method identifies the main directions of walls and is resilient to clutter and partial observations, allowing to extract a clean, abstract geometrical  floor-plan-like description of the environment, which is used to segment, i.e., to identify rooms in, the original occupancy grid map.
This repository should be used with the code provided by [ROSE](https://github.com/tkucner/rose).
For full details refer to [the paper](https://arxiv.org/abs/2203.03519).

## New repository 

We have released a new version of the code integrated with ROS [here](https://github.com/aislabunimi/ROSE2).
In the new version we have also slightly improved the code and fixed some broken dependencies. 
If you want to use the outcome of the method for something new, use the new version of the code.
If you want to replicate the results of the paper, use this code.

## Structure of the repository

1. The folder `RESULTS` contains the results as described in our paper
2. The folder `CODE` contains the source code of our method
3. The folder `MAPFILES` contains the maps we used in our method for evaluation

## Simple working example
Run the main .py file inside the CODE folder, and select the maps you want to process in the DATA folder on the prompt. 

`python runME.py`

If you have some doubts/bugs just write to me and I'll be happy to help.

The code currently published probably needs some  code-level adjustments to fully work. I will try to refactor it and update some stuff in the next future so to make it more ready-to-use. However, it should be pretty straighforward to replicate the results obtained in the `RESULTS` folder. 
In the meantime if you have some issue just write me and I will try promptly answer.

Thanks for your interest in our work!

## Integration with ROS

We are developing an integration of our code with ROS, so that it could be used online by a mobile robot.
The first release is [here](https://github.com/aislabunimi/ROSE2) so refer to this one if you want to use the method with a robot.
