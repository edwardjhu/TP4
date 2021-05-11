SCRIPT_ARGS="--use-cuda \
            --num-workers 8"

COMMON_ARGS="--lin-model \
            --first-order \
            --dataset omniglot \
            --scheduler multistep \
            --batch-size 32 \
            --num-epochs 100 \
            --grad-clip 0.5 \
            --num-ways 5 \
            --num-shots 1 \
            --step-size 0.4 \
            --num-shots-test 1 \
            --normalize None \
            --last-bias-alpha 0 \
            --seed 1"

# Inf-width
SPECIFIC_ARGS="--hidden-size -1 \
            --output-folder ./results/inf \
            --meta-lr 0.1 \
            --sigma1 1 \
            --sigma2 0.03125 \
            --bias-alpha 1"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS

# width = 2
SPECIFIC_ARGS="--hidden-size 2 \
            --output-folder ./results/2 \
            --meta-lr 0.05 \
            --sigma1 0.5 \
            --sigma2 0.5 \
            --bias-alpha 2"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS

# width = 8
SPECIFIC_ARGS="--hidden-size 8 \
            --output-folder ./results/8 \
            --meta-lr 0.1 \
            --sigma1 0.5 \
            --sigma2 0.25 \
            --bias-alpha 1"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS

# width = 32
SPECIFIC_ARGS="--hidden-size 32 \
            --output-folder ./results/32 \
            --meta-lr 0.4 \
            --sigma1 1 \
            --sigma2 0.125 \
            --bias-alpha 0.5"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS

# width = 128
SPECIFIC_ARGS="--hidden-size 128 \
            --output-folder ./results/128 \
            --meta-lr 0.1 \
            --sigma1 1 \
            --sigma2 0.125 \
            --bias-alpha 1"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS

# width = 512
SPECIFIC_ARGS="--hidden-size 512 \
            --output-folder ./results/512 \
            --meta-lr 0.1 \
            --sigma1 1 \
            --sigma2 0.03125 \
            --bias-alpha 1"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS

# width = 2048
SPECIFIC_ARGS="--hidden-size 2048 \
            --output-folder ./results/2048 \
            --meta-lr 0.1 \
            --sigma1 1 \
            --sigma2 0.03125 \
            --bias-alpha 1"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS

# width = 8192
SPECIFIC_ARGS="--hidden-size 8192 \
            --output-folder ./results/8192 \
            --meta-lr 0.1 \
            --sigma1 1 \
            --sigma2 0.03125 \
            --bias-alpha 1"
python meta.train ./data $SCRIPT_ARGS \
                            $COMMON_ARGS \
                            $SPECIFIC_ARGS