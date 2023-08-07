set -e

#CUDA_VISIBLE_DEVICES="1" python calc_el2n.py --use_sgc_embedding --hidden_channels 128

CUDAID=$1
SEED=$2

RSVS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

mkdir logs_s${SEED}/el2n

for (( i=0; i<8; i++ ))
do
  bash scripts/single.sh $CUDAID $SEED ${RSVS[$i]} "el2n"
done
