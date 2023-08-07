set -e

CUDAID=$1
STARTSEED=$2
ENDSEED=$3

for (( i=$STARTSEED; i<$ENDSEED; i++ ))
do
  # FOR MEMO
  #CUDA_VISIBLE_DEVICES="${CUDAID}" python train_on_subsample.py --use_sgc_embedding --seed $i --save_path "/mnt/ogb_datasets/ogbn_papers100M/ckpts" >mem/${i}.out 2>mem/${i}.err
  # FOR EL2N
  #CUDA_VISIBLE_DEVICES="${CUDAID}" python train_on_subsample.py --use_sgc_embedding --seed $i --rsv 1.0 --epochs 10 --hidden_channels 128 --save_path "/mnt/ogb_datasets/ogbn_papers100M/ckpts" >el2n/${i}.out 2>el2n/${i}.err
  # FOR EL2N
  CUDA_VISIBLE_DEVICES="${CUDAID}" python train_on_subsample.py --use_sgc_embedding --seed $i --save_path "/mnt/ogb_datasets/ogbn_papers100M/ckpts" >age/${i}.out 2>age/${i}.err
done
