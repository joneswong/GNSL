set -e

#CUDA_VISIBLE_DEVICES="0" python calc_mem.py --use_sgc_embedding --start_sample_id 0 --end_sample_id 200 &
#CUDA_VISIBLE_DEVICES="1" python calc_mem.py --use_sgc_embedding --start_sample_id 200 --end_sample_id 400 &
#CUDA_VISIBLE_DEVICES="2" python calc_mem.py --use_sgc_embedding --start_sample_id 400 --end_sample_id 600 &
#CUDA_VISIBLE_DEVICES="3" python calc_mem.py --use_sgc_embedding --start_sample_id 600 --end_sample_id 800 &


#python calc_mem.py --use_sgc_embedding --mode 1


CUDAID=$1
SEED=$2

RSVS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

mkdir logs_s${SEED}/mem

for (( i=0; i<8; i++ ))
do
  bash scripts/single.sh $CUDAID $SEED ${RSVS[$i]} "mem"
done


#CUDAID=$1
#RSV0=$2
#RSV1=$3
#
#alphas=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#
#for (( i=0; i<8; i++ ))
#do
#  CUDA_VISIBLE_DEVICES="${CUDAID}" python mlp.py --use_sgc_embedding --rsv $RSV0 --al mem --alpha ${alphas[$i]} >logs/mem/${RSV0}_${alphas[$i]}.out 2>logs/mem/${RSV0}_${alphas[$i]}.err
#done
#
#for (( i=0; i<8; i++ ))
#do
#  CUDA_VISIBLE_DEVICES="${CUDAID}" python mlp.py --use_sgc_embedding --rsv $RSV1 --al mem --alpha ${alphas[$i]} >logs/mem/${RSV1}_${alphas[$i]}.out 2>logs/mem/${RSV1}_${alphas[$i]}.err
#done
