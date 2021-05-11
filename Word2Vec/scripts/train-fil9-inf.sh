#!/bin/bash

wd=${wd:-0.0}
lr=${lr:-0.05}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2
    fi
    shift
done

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/fil9
SAVE_DIR=$DATA_DIR/fil9-vector-inf-wd_$wd-lr_$lr
mkdir $SAVE_DIR
VECTOR_DATA=$SAVE_DIR/fil9-vector-inf.bin

if [ ! -e $TEXT_DATA ]; then
    sh ./create-fil9-data.sh
fi
echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -size 284552 -window 8 -negative 25 \
                                         -sample 1e-4 -threads 110 -binary 1 -min-count 10 -iter 5 -oh 1 -wd $wd -alpha $lr | tee fil9-inf-wd_$wd-lr_$lr.log
