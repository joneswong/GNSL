set -e

CUDA_VISIBLE_DEVICES="0" python train_on_subsample.py --seed 0 --rsv 0.1 --epochs 1 --eval_steps 1 &
CUDA_VISIBLE_DEVICES="1" python train_on_subsample.py --seed 1 --rsv 0.1 --epochs 1 --eval_steps 1 &
