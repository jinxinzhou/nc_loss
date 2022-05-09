#!/bin/bash
#SBATCH --account=qingqu1
#SBATCH --job-name=fl_1_1
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=fl_1_1_%j.out

# module purge
# module load python3.7-anaconda/2020.02
# # module load cuda/10.2.89
eval "$(conda shell.bash hook)"
conda activate /scratch/qingqu_root/qingqu1/jinxinz/DL/ImgMLP


# FL width=1
python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs800-1/ --seed 1 --width 1 -e 800 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs800-1/ --seed 1 --width 1 -e 800 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs400-1/ --seed 1 --width 1 -e 400 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs400-1/ --seed 1 --width 1 -e 400 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs200-1/ --seed 1 --width 1 -e 200 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs200-1/ --seed 1 --width 1 -e 200 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs100-1/ --seed 1 --width 1 -e 100 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-1-cifar10-fl-epochs100-1/ --seed 1 --width 1 -e 100 --model wide_resnet50 --dataset cifar10



# FL width=0.5
python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs800-1/ --seed 1 --width 0.5 -e 800 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs800-1/ --seed 1 --width 0.5 -e 800 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs400-1/ --seed 1 --width 0.5 -e 400 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs400-1/ --seed 1 --width 0.5 -e 400 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs200-1/ --seed 1 --width 0.5 -e 200 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs200-1/ --seed 1 --width 0.5 -e 200 --model wide_resnet50 --dataset cifar10

python train.py --bias --loss focal_loss --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs100-1/ --seed 1 --width 0.5 -e 100 --model wide_resnet50 --dataset cifar10 --gamma 3
python evaluate.py --bias --save-path ./res50_cifar10/wideres50-05-cifar10-fl-epochs100-1/ --seed 1 --width 0.5 -e 100 --model wide_resnet50 --dataset cifar10





