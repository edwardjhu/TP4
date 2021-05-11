SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec compute-accuracy

word2vec : word2vec.c
	$(CC) word2vec.c -o ${BIN_DIR}/word2vec $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/compute-accuracy $(CFLAGS)
	chmod +x ${SCRIPTS_DIR}/*.sh

clean:
	pushd ${BIN_DIR} && rm -rf word2vec compute-accuracy; popd
