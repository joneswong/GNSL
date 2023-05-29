set -e

CUDAID=$1
ratio=$2


CUDA_VISIBLE_DEVICES="$CUDAID" python mlp.py --rsv $ratio --use_sgc_embedding >logs/random/${ratio}.out 2>logs/random/${ratio}.err
