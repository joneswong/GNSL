set -e

DID=$1
SEED=$2
AL=$3

ratios=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

mkdir logs_s${SEED}/${AL}

for (( i=0; i<8; i++ ))
do
  CUDA_VISIBLE_DEVICES="$DID" python train_sage.py --seed $SEED --al $AL --rsv ${ratios[$i]} --alpha ${ratios[$i]} >logs_s${SEED}/${AL}/${ratios[$i]}_${ratios[$i]}.out 2>logs_s${SEED}/${AL}/${ratios[$i]}_${ratios[$i]}.err
done
