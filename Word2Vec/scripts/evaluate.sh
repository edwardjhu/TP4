#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

FOLDER=$1


for f in $FOLDER/*
do
	if [[ "$f" == *"eval"* ]]; then
        continue
    fi 
    if [ -e "${f}.eval" ]; then
        continue
    fi
    echo $f
	$BIN_DIR/compute-accuracy $f 0 < $DATA_DIR/questions-words.txt | tee "${f}.eval"
done

