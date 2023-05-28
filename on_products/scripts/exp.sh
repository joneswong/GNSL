set -e

ratios=(0.0625 0.0575 0.055 0.0525 0.0475 0.045 0.0425 0.03)

for (( i=0; i<8; i++ ))
do
  CUDA_VISIBLE_DEVICES="${i}" python main.py --rsv ${ratios[$i]} >logs/${ratios[$i]}.out 2>logs/${ratios[$i]}.err &
done
