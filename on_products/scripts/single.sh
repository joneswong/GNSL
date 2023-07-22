set -e

SEED=$1
AL=$2

ratios=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

mkdir logs/$AL

for (( i=0; i<8; i++ ))
do
  CUDA_VISIBLE_DEVICES="$i" python train_gcn.py --runs 3 --seed $SEED --al $AL --rsv ${ratios[$i]} --alpha ${ratios[$i]} >logs/$AL/${ratios[$i]}_${ratios[$i]}.out 2>logs/$AL/${ratios[$i]}_${ratios[$i]}.err &
done
