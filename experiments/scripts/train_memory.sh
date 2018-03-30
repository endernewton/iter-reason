#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
NET_BASE=res50

GPU_ID=$1
DATASET=$2
NET=${NET_BASE}_$3

OIFS=$IFS
IFS='a'
STEP="$4"
STEPSIZE="["
for i in $STEP; do
  STEPSIZE=${STEPSIZE}"${i}0000,"
done
STEPSIZE=${STEPSIZE}"]"
IFS=$OIFS

ITERS=${5}0000

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    VAL_IMDB="coco_2014_minival"
    ;;
  vg)
    TRAIN_IMDB="visual_genome_train_5"
    VAL_IMDB="visual_genome_val_5"
    ;;
  ade)
    TRAIN_IMDB="ade_train_5"
    VAL_IMDB="ade_mval_5"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  EXTRA_ARGS_SLUG=${EXTRA_ARGS_SLUG}_${4}_${5}
else
  EXTRA_ARGS_SLUG=${4}_${5}
fi

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_iter_${ITERS}.ckpt
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_memory.py \
    --weight data/imagenet_weights/${NET_BASE}.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${VAL_IMDB} \
    --iters ${ITERS} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
fi

./experiments/scripts/test_memory.sh $@
