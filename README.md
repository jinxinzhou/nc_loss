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

# Kangning:
## MSE: width=[8, 16]; epochs=[100, 200, 400, 800]; seed=[1, 123, 321]
~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs100-1/ 1 8 100
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs100-123/ 123 8 100
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs100-321/ 321 8 100
~~~

~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs200-1/ 1 8 200
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs200-123/ 123 8 200
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs200-321/ 321 8 200
~~~

~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs400-1/ 1 8 400
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs400-123/ 123 8 400
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs400-321/ 321 8 400
~~~

~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs800-1/ 1 8 800
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs800-123/ 123 8 800
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-8-cifar100-mse-epochs800-321/ 321 8 800
~~~

~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs100-1/ 1 16 100
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs100-123/ 123 16 100
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs100-321/ 321 16 100
~~~

~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs200-1/ 1 16 200
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs200-123/ 123 16 200
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs200-321/ 321 16 200
~~~

~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs400-1/ 1 16 400
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs400-123/ 123 16 400
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs400-321/ 321 16 400
~~~

~~~python
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs800-1/ 1 16 800
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs800-123/ 123 16 800
$ sbatch cmds-wide.sh mean_square_error ./checkpoints/wideres26-16-cifar100-mse-epochs800-321/ 321 16 800
~~~

## Label Smooth: width=[8, 16]; epochs=[100, 200, 400, 800]; seed=[1, 123, 321]
~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs100-1/ 1 8 100
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs100-123/ 123 8 100
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs100-321/ 321 8 100
~~~

~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs200-1/ 1 8 200
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs200-123/ 123 8 200
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs200-321/ 321 8 200
~~~

~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs400-1/ 1 8 400
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs400-123/ 123 8 400
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs400-321/ 321 8 400
~~~

~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs800-1/ 1 8 800
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs800-123/ 123 8 800
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-8-cifar100-ls-epochs800-321/ 321 8 800
~~~

~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs100-1/ 1 16 100
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs100-123/ 123 16 100
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs100-321/ 321 16 100
~~~

~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs200-1/ 1 16 200
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs200-123/ 123 16 200
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs200-321/ 321 16 200
~~~

~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs400-1/ 1 16 400
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs400-123/ 123 16 400
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs400-321/ 321 16 400
~~~

~~~python
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs800-1/ 1 16 800
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs800-123/ 123 16 800
$ sbatch cmds-wide.sh label_smoothing ./checkpoints/wideres26-16-cifar100-ls-epochs800-321/ 321 16 800
~~~

# Sheng:
## CE: width=[8, 16]; epochs=[100, 200, 400, 800]; seed=[1, 123, 321]
~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs100-1/ 1 8 100
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs100-123/ 123 8 100
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs100-321/ 321 8 100
~~~

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs200-1/ 1 8 200
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs200-123/ 123 8 200
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs200-321/ 321 8 200
~~~

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs400-1/ 1 8 400
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs400-123/ 123 8 400
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs400-321/ 321 8 400
~~~

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs800-1/ 1 8 800
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs800-123/ 123 8 800
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-8-cifar100-ce-epochs800-321/ 321 8 800
~~~

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs100-1/ 1 16 100
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs100-123/ 123 16 100
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs100-321/ 321 16 100
~~~

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs200-1/ 1 16 200
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs200-123/ 123 16 200
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs200-321/ 321 16 200
~~~

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs400-1/ 1 16 400
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs400-123/ 123 16 400
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs400-321/ 321 16 400
~~~

~~~python
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs800-1/ 1 16 800
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs800-123/ 123 16 800
$ sbatch cmds-wide.sh cross_entropy ./checkpoints/wideres26-16-cifar100-ce-epochs800-321/ 321 16 800
~~~

## Focal loss: width=[8, 16]; epochs=[100, 200, 400, 800]; seed=[1, 123, 321]
~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs100-1/ 1 8 100
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs100-123/ 123 8 100
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs100-321/ 321 8 100
~~~

~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs200-1/ 1 8 200
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs200-123/ 123 8 200
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs200-321/ 321 8 200
~~~

~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs400-1/ 1 8 400
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs400-123/ 123 8 400
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs400-321/ 321 8 400
~~~

~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs800-1/ 1 8 800
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs800-123/ 123 8 800
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-8-cifar100-fl-epochs800-321/ 321 8 800
~~~

~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs100-1/ 1 16 100
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs100-123/ 123 16 100
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs100-321/ 321 16 100
~~~

~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs200-1/ 1 16 200
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs200-123/ 123 16 200
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs200-321/ 321 16 200
~~~

~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs400-1/ 1 16 400
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs400-123/ 123 16 400
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs400-321/ 321 16 400
~~~

~~~python
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs800-1/ 1 16 800
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs800-123/ 123 16 800
$ sbatch cmds-wide.sh focal_loss ./checkpoints/wideres26-16-cifar100-fl-epochs800-321/ 321 16 800
~~~


