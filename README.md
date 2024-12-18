# Task Prompt Guided Network for Remote Sensing Image Salient Object Detection: A Multi-task Learning Perspective

Welcome to the official repository for the paper "Task Prompt Guided Network for Remote Sensing Image Salient Object Detection: A Multi-task Learning Perspective"

### The Initialization Weights for Training
Download pre-trained classification weights of the [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) , and place the ` .pth ` files in ` ./pretrained_model ` directory. These weights are essential for initializing the model during training.

### Trained Weights of MTPNet for Testing

[Download](https://pan.baidu.com/s/1It1POLIDvCxVIaY0i7aSuw&pwd=axtm)

### Train
Please download the pre-trained model weights and dataset first. Next, generate the path of the training set and the test set, and change the dataset path in the code to the path of the dataset you specified.

~~~python
python train.py
~~~

### Test
Download the MTPNet model weights, create the necessary directories to store these files, and be sure to update the corresponding paths in the code accordingly. 

~~~python

python test.py

~~~
