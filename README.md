# CV Final Project

## Introduction

We propose a method for generating animated images of numbers using generative adversarial networks and reinforcement learning. Our approach involves training a generative agent to control a simulated painting environment, with rewards provided by a discriminator network that is simultaneously trained to evaluate the authenticity of the agent's samples. By training on a dataset of images and labels, our model is able to extract the features of the images and generate Bessel curves that represent the strokes used to draw the image. The generated stroke data is then drawn frame by frame on a drawing board, resulting in an animation of the number being drawn. Our model has potential applications in creative and artistic contexts.

## Setup Environment

Our project depends on Deepmind implmentation of libmypaint, which is a library for simulating a drawing board. To install the library. For your convinience, we provide a Dockerfile for building a docker container which includes all the dependencies.

Build the docker image and run the container:

```
cd env
docker build -t cv_final .
docker run --rm -it --gpus 1 cv_final
```

## How to Run Our Code

1. You need to download MNIST (.png) dataset and put it in the correct directory (see details in image_loader.py).
2. Change parameters and libmypaint path in config.py.
3. Run main.py to train the model.
4. Run demo.py to generate animation.

## Results

![1](./results/merged%201.gif)
![2](./results/merged%202.gif)
![3](./results/merged%203.gif)
![4](./results/merged%204.gif)
![5](./results/merged%205.gif)
