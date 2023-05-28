set -e

CUDAID=$1
STARTSEED=$2
ENDSEED=$3

for (( i=$STARTSEED; i<$ENDSEED; i++ ))
do
  # FOR MEMO
  #CUDA_VISIBLE_DEVICES="${CUDAID}" python train_on_subsample.py --seed $i --rsv 0.7 --eval_steps 5 >temp/${i}.out 2>temp/${i}.err
  # FOR EL2N
  CUDA_VISIBLE_DEVICES="${CUDAID}" python train_on_subsample.py --seed $i --epochs 5 --hidden_channels 128 --begin_valid_at 5 --eval_steps 1 --save_path en2l >temp/${i}.out 2>temp/${i}.err
done
