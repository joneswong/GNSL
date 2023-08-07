set -e

#CUDA_VISIBLE_DEVICES="1" python calc_mem.py --use_sgc_embedding --mode 1 --infl "max"

CUDAID=$1
SEED=$2

RSVS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for (( i=0; i<8; i++ ))
do
  bash scripts/single.sh $CUDAID $SEED ${RSVS[$i]} "infl-max"
done
