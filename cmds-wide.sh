 #!/bin/bash
 #SBATCH --account=qingqu1
 #SBATCH --job-name=Nc
 #SBATCH --nodes=1
 #SBATCH --time=48:00:00
 #SBATCH --mem=8GB
 #SBATCH --partition=gpu
 #SBATCH --gres=gpu:1
 #SBATCH --output=NC_%j.out

 module purge
 module load python3.7-anaconda/2020.02
 # module load cuda/10.2.89
 eval "$(conda shell.bash hook)"
 conda activate /scratch/qingqu_root/qingqu1/jinxinz/DL/ImgMLP

 loss=$1
 save_path=$2
 seed=$3
 width=$4
 epochs=$5

 python train.py --bias --loss $loss --save-path $save_path --seed $seed --width $width -e $epochs
 python evaluate.py --bias --save-path $save_path --seed $seed --width $width -e $epochs


