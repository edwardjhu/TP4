Meta-Learning/Few-Shot Learning Task in "Feature Learning in Infinite-Width Neural Networks"
========

This repo contains code to replicate the MAML experiments in the paper *Feature Learning in Infinite-Width Neural Networks*.

Our code is based on a fork of [pytorch-maml](https://github.com/tristandeleu/pytorch-maml) with our implementation of NTK, GP and MUP networks.

The important files are structured as follows:
```
+-- inf                              ()
|   +-- dynamicarray.py              (dynamic array used for GP/NTK implementation)
|   +-- inf1lp.py                    (implements GP and NTK for MAML)
|   +-- optim.py                     (optimizers for inf-width nets)
+-- meta                             (A fork of "pytorch-maml")
|   +-- maml/                        ()
|   |   +-- datasets.py              (datasets)
|   |   +-- model.py                 (finite nets and inf-width linear net)
|   |   +-- utils.py                 (utils from "pytorch-maml")
|   |   +-- metalearners/            ()
|   |   |   +-- maml.py              (MAML impl. for finite nets and inf-width linear net)
|   |   |   +-- infmaml.py           (MAML impl. for GP/NTK)
|   |   |   +-- meta_sgd.py          (SGD for MAML from "pytorch-maml")
|   +-- train.py                     (Training a model)
|   +-- test.py                      (Evaluating a trained model)
```

## Reproducing Our Results

### Training

We trained and tested finite-width linear networks with varying widths and their GP/NTK/MUP infinite-width limits.

We tune the following hyperparameters, which are described in the paper:

$\sigma_u$, $\sigma_v$, $\eta$, $\alpha$, $\sigma_b$ (only tuned for NTK/GP)

We will refer to them as `$sigma_u`, `$sigma_v`, `$sigma_v`, `$eta`, `$alpha` and `$sigma_b` in the code snippet.
The hyperparameters swept, and eventually used, are in Section D.1 and Table 4.

To train a finite-width linear model with $width={2, 8, 32, 128, 512, 2048, 8192} on GPUs, run:
```
TRAINING_ARGS="--hidden-size ${width} \
               --lin-model \  
               --sigma1 ${sigma_u} \
               --sigma2 ${sigma_v} \
               --bias-alpha ${alpha} \
               --meta-lr ${eta}"

COMMON_ARGS="--use-cuda \
             --first-order \
             --num-workers 8
             --dataset omniglot \
             --scheduler multistep \
             --batch-size 32 \
             --num-epochs 100 \
             --grad-clip 0.5 \
             --num-ways 5 \
             --num-shots 1 \
             --step-size 0.4
             --num-shots-test 1 \
             --normalize None"

python meta.train ./data $COMMON_ARGS \
                         $TRAINING_ARGS
```

To train the MUP infinite-width limit, swap `TRAINING_ARGS` with:
```
TRAINING_ARGS="--lin-model \
               --hidden-size -1 \
               --sigma1 ${sigma_u} \
               --sigma2 ${sigma_v} \
               --bias-alpha ${alpha} \
               --meta-lr ${eta}"
```

To train the NTK infinite-width limit, swap `TRAINING_ARGS` with:
```
TRAINING_ARGS="--ntk1lp \
               --hidden-size -1 \
               --sigma1 ${sigma_u} \
               --sigma2 ${sigma_v} \
               --sigmab ${sigma_b} \
               --bias-alpha ${alpha} \
               --meta-lr ${eta}"
```

To train the GP infinite-width limit, swap `TRAINING_ARGS` with:
```
TRAINING_ARGS="--gp1lp \
               --hidden-size -1 \
               --sigma1 ${sigma_u} \
               --sigma2 ${sigma_v} \
               --sigmab ${sigma_b} \
               --bias-alpha ${alpha} \
               --meta-lr ${eta}"
```

### Evaluation

To evaluate a train model under $model_dir on the test set, run:
```
python meta.test $model_dir/config.json --folder ./data --verbose --use-cuda --seed 1 --num-worker 8 --num-steps 20
```
