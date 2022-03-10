# Are All Losses Created Equal?

## Measuring NC during network training

### Datasets

By default, the code assumes the datasets for CIFAR100 are stored under `~/data/`. If the datasets are not there, they will be automatically downloaded from `torchvision.datasets`. User may change this default location of datasets in `args.py` through the argument `--dataset-root`.

### Training with different losses, seeds, witdths and epochs
#### loss is chosed from one of [cross_entropy, mean_square_error, focal_loss, label_smoothing]. The default is set to be "cross_entropy";
#### seed is chosed from one of [1, 123, 321]. The default is set to be "1";
#### width is chosed from one of [1, 2, 4, 8, 16]. The default is set to be "4";
#### epoch is chosed from one of [100, 200, 400, 800]. The default is set to be "400".
Additional, I set the args.save-path as "./checkpoints/wideres26-width-cifar100-loss-epochs#-seed/" format, such as "./checkpoints/wideres26-8-cifar100-ce-epochs400-1/" and "./checkpoints/wideres26-4-cifar100-fl-epochs800-123/". 

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs400-1/ 1 8 400
~~~
