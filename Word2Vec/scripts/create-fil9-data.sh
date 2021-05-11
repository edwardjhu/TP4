#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/fil9
ZIPPED_TEXT_DATA="$DATA_DIR/enwik9.gz"

if [ ! -e $DATA_DIR ]; then
  mkdir $DATA_DIR
fi

if [ ! -e $TEXT_DATA ]; then
  if [ ! -e $ZIPPED_TEXT_DATA ]; then
    wget http://mattmahoney.net/dc/enwik9.zip -O $ZIPPED_TEXT_DATA
	fi
	gzip -d $ZIPPED_TEXT_DATA -f
	perl wikifil.pl $DATA_DIR/enwik9 > $TEXT_DATA
fi
