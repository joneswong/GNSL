set -e

CUDAID=$1
SEED=$2
RSV=$3
AL=$4

CUDA_VISIBLE_DEVICES="${CUDAID}" python mlp.py --use_sgc_embedding --runs 5 --seed $SEED --rsv $RSV --alpha $RSV --al $AL >logs_s${SEED}/${AL}/${RSV}_${RSV}.out 2>logs_s${SEED}/${AL}/${RSV}_${RSV}.err
