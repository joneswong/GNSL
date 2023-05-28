set -e

CUDAID=$1
STARTSEED=$2
ENDSEED=$3

CUDA_VISIBLE_DEVICES="${CUDAID}" python calc_mem.py --start_sample_id $STARTSEED --end_sample_id $ENDSEED >temp/${CUDAID}.out 2>temp/${CUDAID}.err
