#!/bin/bash
SCRIPT="src/main.py"
CUDA_DEVICE=$1

if [[ $CUDA_DEVICE == "" ]]; then
    CUDA_TAGS=""
else
    CUDA_TAGS="--cuda --cuda_device $CUDA_DEVICE"
fi

run() {
    for i in {1..5}; do
        seed=$((42 + $i))
        name="${1}"
        synth_dim=$2
        power_one_prob=$3
        power_scalar=$4
        big_ct=10000
        train_cmd="python -u $SCRIPT \
            --name $name \
            --mode train \
            --preprocess_mode synth \
            --synth_mode power \
            --synth_train_ct $big_ct \
            --synth_dev_ct 10000 \
            --synth_test_ct $big_ct \
            --synth_seed $seed \
            --synth_dim $synth_dim \
            --power_one_prob $power_one_prob \
            --power_scalar $power_scalar \
            --epochs 600 \
            --batch_size $big_ct \
            $CUDA_TAGS \
            $@"
        $train_cmd
    done
}


p1_list="
0.20
0.18
0.16
0.14
0.12
0.1
0.08
0.06
0.04
0.02
"

dim=10
decay_flags="--lr_decay_type linear --lr_decay_min 0.001"

exp () {
    ps=$1
    for p1 in $p1_list; do
        run normal $dim $p1 $ps --lr 0.1 --momentum 0.9 "$decay_flags"
        run balanced $dim $p1 $ps --loss_type class_weighted_ce --lr 0.1 --momentum 0.9 "$decay_flags"
        for alpha in 0.001 0.01 0.05; do
           run cvar_$alpha $dim $p1 $ps --loss_type cvar --lr 0.1 --cvar_alpha $alpha "$decay_flags"
        done
        for temp in 0.8 1.0 1.2; do
           run hcvar_$temp $dim $p1 $ps --loss_type hcvar --lr 0.1 --hcvar_alpha 0.01 --hcvar_temp $temp "$decay_flags"
        done
    done
}

exp 1
