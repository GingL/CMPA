#!/bin/bash
# custom config
DATA=/path/to/data
TRAINER=CMPA

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep20_batch4_16ctx
SHOTS=16

for SEED in 1 2 3
do
    DIR=output/crossdata/${DATASET}/${SHOTS}shots/seed${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/16shots/seed${SEED}\
    --load-epoch 5 \
    --eval-only
done