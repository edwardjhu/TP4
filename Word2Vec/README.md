Word2Vec Experiment in "Feature Learning in Infinite-width Neural Networks"
========

This repo contains code to replicate the Word2Vec experiments in the paper *Feature Learning in Infinite-Width Neural Networks*.

Adapted from https://code.google.com/p/word2vec/

We made the following changes for the purpose of our experiment.
1. Simplified training options to keep just CBOW with negative examples;
2. Added the option to train an infinite-width $\mu$P network;
3. Added loss logging and thread synchronizations;
4. Added multi-threaded evaluation on the word analogy task.

The important files are structured as follows:
```
+-- scripts                          (Training/data scripts)
|   +-- create-{text8,fil9}-data.sh  (Download and prepare {text8,fil9})
|   +-- train-{text8,fil9}.sh        (Train a finite-width network on {text8,fil9})
|   +-- train-{text8,fil9}-inf.sh    (Train a infinite-width network on {text8,fil9})
|   +-- evalute.sh                   (Evaluate a trained model on the word-analogy task)
+-- src                              (Source files)
|   +-- word2vec.c                   (The main training script)
|   +-- compute-accuracy.c           (The evaluation script)
```
To reproduce our result using text8 as an example:

First, we compile the code and switch to directory `scripts`
```
mkdir bin; cd src; make
cd ../scripts
```
    
Using width={64, 256, 1024}, we train finite-networks with the command (the data will be downloaded when run for the first time)
    
`bash train-text8.sh --width $width --wd 0.001 --lr 0.05`

We can also train an infinite-width network with the same hyperparameters

`bash train-text8-inf.sh --wd 0.001 --lr 0.05`

Finally, to evaluate the checkpoints (width={64, 256, 1024, inf})

`bash evaluate.sh ../data/text8-vector-$width-wd_0.001-lr_0.05`

The result can be retrieved in the folder that contains the checkpoints


Note that this CPU-based implementation inevitably runs into race conditions, especially when using many threads.
The result might differ numerically when run on machines with different CPUs, but the overall trend should hold as long as we use the same machine for a given dataset.

In particular, our text8 result is produced on an Intel Xeon Platinum 8272CL with 72 virtual cores, and our fil9 result on an AMD EPYC 7V12 CPU with 120 virtual cores.
