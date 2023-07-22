set -e

CUDAID=$1
RSV0=$2
#RSV1=$3

alphas=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for (( i=0; i<8; i++ ))
do
  CUDA_VISIBLE_DEVICES="${CUDAID}" python main.py --rsv $RSV0 --al "age" --alpha ${alphas[$i]} >logs/age/${RSV0}_${alphas[$i]}.out 2>logs/age/${RSV0}_${alphas[$i]}.err
done
