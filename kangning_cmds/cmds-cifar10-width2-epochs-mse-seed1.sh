#!/bin/bash
#SBATCH --account=qingqu1
#SBATCH --job-name=mse_2
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --mem=8GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mse_2_%j.out

# module purge
# module load python3.7-anaconda/2020.02
# # module load cuda/10.2.89
eval "$(conda shell.bash hook)"
conda activate /scratch/qingqu_root/qingqu1/jinxinz/DL/ImgMLP


# MSE width=2
python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs400-1/ --seed 1 --width 2 -e 400 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs400-1/ --seed 1 --width 2 -e 400 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs200-1/ --seed 1 --width 2 -e 200 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs200-1/ --seed 1 --width 2 -e 200 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs100-1/ --seed 1 --width 2 -e 100 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-mse-epochs100-1/ --seed 1 --width 2 -e 100 --model wide_resnet50 --dataset cifar10


# MSE width=0.25
python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs800-1/ --seed 1 --width 0.25 -e 800 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs800-1/ --seed 1 --width 0.25 -e 800 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs400-1/ --seed 1 --width 0.25 -e 400 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs400-1/ --seed 1 --width 0.25 -e 400 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs200-1/ --seed 1 --width 0.25 -e 200 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs200-1/ --seed 1 --width 0.25 -e 200 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss mean_square_error --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs100-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar10
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-mse-epochs100-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar10





