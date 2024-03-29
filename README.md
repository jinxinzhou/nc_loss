# Are All Losses Created Equal? A Neural Collapse Perspective

This is the code for the [paper](https://neurips.cc/virtual/2022/poster/53974) "Are All Losses Created Equal: A Neural Collapse Perspective".

Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), 2022

## Introduction

- We provide the first global optimization landscape analysis of *Neural Collapse* (NC) under focal loss (FL) and label smoothing (LS) loss – an intriguing empirical phenomenon that arises in the last-layer classifiers and features of neural networks during the terminal phase of training. Besides, We show that a family of losses exihibts NC once the training reachs the terminal phase.
- we emprically find that all losses (i.e., CE, LS, FL, MSE) lead to largely identical features on training data by large DNNs and sufficiently many iterations.
- we emprically find that all losses (CE, LS, FL, MSE) lead to largely identical performance on test data by large DNNs and sufficiently many iterations.

## Environment

- CUDA 11.0
- python 3.8.3
- torch 1.6.0
- torchvision 0.7.0
- scipy 1.5.2
- numpy 1.19.1

## Measuring NC during network training

### Datasets

By default, the code assumes the datasets for CIFAR10 and CIFAR100 are stored under `../data/`. If the datasets are not there, they will be automatically downloaded from `torchvision.datasets`. User may change this default location of datasets in `args.py` through the argument `--dataset-root`.

### Training with different losses, seeds, witdths and epochs
#### loss is chosed from one of [cross_entropy, mean_square_error, focal_loss, label_smoothing]. The default is set to be "cross_entropy";
#### seed is chosed from one of [1, 123, 321]. The default is set to be "1";
#### width is chosed from one of [0.25, 0.5, 1, 2]. The default is set to be "2";
#### epoch is chosed from one of [100, 200, 400, 800]. The default is set to be "800".
Additional, I set the args.save-path as "./wideres50_cifar100/wideres50-width-loss-epochs-seed/" format, such as "./wideres50_cifar100/wideres50-2-ce-800-1/" and "./wideres50_cifar100/wideres50-2-mse-800-123/". 

## Some training examples on CIFAR100: 
~~~python
$ python train.py --bias --loss cross_entropy --save-path ./wideres50_cifar100/wideres50-2-ce-800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar100
$ python train.py --bias --loss mean_square_error --save-path ./wideres50_cifar100/wideres50-1-mse-400-123/ --seed 123 --width 1 -e 400 --model wide_resnet50 --dataset cifar100
$ python train.py --bias --loss focal_loss --save-path ./wideres50_cifar100/wideres50-05-fl-200-321/ --seed 321 --width 0.5 -e 200 --model wide_resnet50 --dataset cifar100
$ python train.py --bias --loss label_smoothing --save-path ./wideres50_cifar100/wideres50-025-ls-100-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar100
~~~

## Some evaluate examples on CIFAR100: 
~~~python
$ python evaluate.py --bias --save-path ./wideres50_cifar100/wideres50-2-ce-800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar100
$ python evaluate.py --bias --save-path ./wideres50_cifar100/wideres50-2-mse-800-123/ --seed 123 --width 1 -e 400 --model wide_resnet50 --dataset cifar100
$ python evaluate.py --bias --save-path ./wideres50_cifar100/wideres50-2-fl-800-321/ --seed 321 --width 0.5 -e 200 --model wide_resnet50 --dataset cifar100
$ python evaluate.py --bias --save-path ./wideres50_cifar100/wideres50-2-ls-800-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar100
~~~

## Some training examples on CIFAR10: 
~~~python
$ python train.py --bias --loss cross_entropy --save-path ./wideres50_cifar10/wideres50-2-ce-800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar10
$ python train.py --bias --loss mean_square_error --save-path ./wideres50_cifar10/wideres50-1-mse-400-2/ --seed 2 --width 1 -e 400 --model wide_resnet50 --dataset cifar10
$ python train.py --bias --loss focal_loss --save-path ./wideres50_cifar10/wideres50-05-fl-200-3/ --seed 3 --width 0.5 -e 200 --model wide_resnet50 --dataset cifar10
$ python train.py --bias --loss label_smoothing --save-path ./wideres50_cifar10/wideres50-025-ls-100-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar10
~~~

## Some evaluate examples on CIFAR10: 
~~~python
$ python evaluate.py --bias --save-path ./wideres50_cifar10/wideres50-2-ce-800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar10
$ python evaluate.py --bias --save-path ./wideres50_cifar10/wideres50-2-mse-800-2/ --seed 2 --width 1 -e 400 --model wide_resnet50 --dataset cifar10
$ python evaluate.py --bias --save-path ./wideres50_cifar10/wideres50-2-fl-800-3/ --seed 3 --width 0.5 -e 200 --model wide_resnet50 --dataset cifar10
$ python evaluate.py --bias --save-path ./wideres50_cifar10/wideres50-2-ls-800-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar10
~~~

## Plot the figures in paper
Finally, the figures in the paper can be reproduced as following:

1.The experiment of the dynamic of neural collapse metrics can be visualized by plotting them in figures:
~~~python
$ python plot.py
~~~

2.The experiment of the heatmap of test accuracy can be visualized by plotting them in figures:

~~~python
$ python plot_heatmap.py
~~~

## Citation and reference 
For technical details and full experimental results, please check [our paper](https://arxiv.org/pdf/2210.02192.pdf).
```
@article{zhou2022all,
  title={Are All Losses Created Equal: A Neural Collapse Perspective},
  author={Zhou, Jinxin and You, Chong and Li, Xiao and Liu, Kangning and Liu, Sheng and Qu, Qing and Zhu, Zhihui},
  journal={arXiv preprint arXiv:2210.02192},
  year={2022}
}
```
