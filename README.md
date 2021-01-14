# Deep-Learning-with-added-landmarks-for-improved-object-segmentation
Research project in Computational Science, IT department, Uppsala University.

## Context
The aim of this project is to assess adding landmarks to two variations of the U-Net model on the segmentation of abdominal organs.

# Content
This repository contains:
- scripts for the 2D model in *2D_network/*
- scripts for the 3D model in *3D_network/*
- scripts to train and evaluate the U-Net with Multi-task learning in *MultiTaskLearning/*


## Usage
### 2D network
To train and evaluate the 2D model without adding the landmarks:
`python train_vtk.py --batchsize --lr --epochs --organ --load`

```shell script
> python train_vtk.py

Segment organ from input image.

arguments:
  --epochs int,      Number of epochs, default=20
  --batchsize int,   Batch size, default=10
  --lr int,          Learning rate, default=0.00001
  --load, FILE       Specify the name of the model to evaluate, default=CP_epoch20.pth
  --organ, str       Specify the organ to segment, default=liver
  --no-train         Do not train the model, only evaluate it 
```

To train and evaluate the 2D model with early fusion of the landmarks:
`python train_vtk_ef.py --batchsize --lr --epochs --organ --load`

To train and evaluate the 2D model with late fusion of the landmarks:
`python train_vtk_lf.py --batchsize --lr --epochs --organ --load`


### Theory
- The U-Net (2D model) architecture
<p align="center">   
<img src="https://drive.google.com/uc?export=view&id=19m35nBOGW2TEOq5O4b9oZU9uT0dyoDRJ" width="650" height="450">
</p>

- The multi-task learning model 
<p align="center">   
<img src="https://drive.google.com/uc?export=view&id=1A987F0WKnTg9I7gX-SRzQEoIR1PBMSIL" width="550" height="450">
</p>

## References
- Olaf Ronneberger, Philipp Fischer, and Thomas Brox. “U-Net: Convolutional Networks for
Biomedical Image Segmentation”. In: CoRR abs/1505.04597 (2015). arXiv: 1505.04597.
url: http://arxiv.org/abs/1505.04597.
- Thi work was inspired by the code in https://github.com/milesial/Pytorch-UNet.
