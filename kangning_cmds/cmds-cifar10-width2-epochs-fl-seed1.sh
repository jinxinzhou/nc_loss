#!/bin/bash
#SBATCH --account=qingqu1
#SBATCH --job-name=fl_2_1
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --mem=8GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=fl_2_1_%j.out

# module purge
# module load python3.7-anaconda/2020.02
# # module load cuda/10.2.89
eval "$(conda shell.bash hook)"
conda activate /scratch/qingqu_root/qingqu1/jinxinz/DL/ImgMLP

# FL width=2
python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs800-1/ --seed 1 --width 2 -e 800 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs400-1/ --seed 1 --width 2 -e 400 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs400-1/ --seed 1 --width 2 -e 400 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs200-1/ --seed 1 --width 2 -e 200 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs200-1/ --seed 1 --width 2 -e 200 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs100-1/ --seed 1 --width 2 -e 100 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-2-cifar10-fl-epochs100-1/ --seed 1 --width 2 -e 100 --model wide_resnet50 --dataset cifar10


# FL width=0.25
python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs800-1/ --seed 1 --width 0.25 -e 800 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs800-1/ --seed 1 --width 0.25 -e 800 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs400-1/ --seed 1 --width 0.25 -e 400 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs400-1/ --seed 1 --width 0.25 -e 400 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs200-1/ --seed 1 --width 0.25 -e 200 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs200-1/ --seed 1 --width 0.25 -e 200 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs100-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-025-cifar10-fl-epochs100-1/ --seed 1 --width 0.25 -e 100 --model wide_resnet50 --dataset cifar10





