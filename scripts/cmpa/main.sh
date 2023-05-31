#!/bin/bash

# custom config
DATA=/path/to/data
TRAINER=CMPA

DATASET=$1
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
CFG=vit_b16_c2_ep20_batch4_16ctx  # config file
NCTX=16  # number of context tokens
SUBSAMPLES=all # all, base, new, ten
PROMPT_DEPTH=9 # prompt depth
FUSING=mean # mean, first, max
PS=True # parameter sharing, True or False

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${SHOTS}shots/seed${SEED}
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.CMPA.N_CTX ${NCTX} \
    TRAINER.CMPA.PROMPT_DEPTH ${PROMPT_DEPTH} \
    TRAINER.CMPA.FUSING ${FUSING} \
    TRAINER.CMPA.PS ${PS} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLES}
done
