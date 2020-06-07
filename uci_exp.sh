#!/bin/bash
SCRIPT="src/main.py"
UCI_DIR="uci_data"

if [[ $CUDA_DEVICE == "" ]]; then
    CUDA_TAGS=""
else
    CUDA_TAGS="--cuda --cuda_device $CUDA_DEVICE"
fi
run() {
    name=$1
    preproc=$2
    train_path=$3
    test_path=$4
    train_cmd="python -u $SCRIPT --mode train --train_path $train_path --dev_path $test_path --test_path $test_path --preprocess_mode $preproc --name "${name}_${preproc}" $CUDA_TAGS $@"
    $train_cmd
}


make_dataset () {
    name=$1
    echo "$name $UCI_DIR/$name/$name.train $UCI_DIR/$name/$name.test"
}

lr_decay_flags="--lr_decay_type linear --lr_decay_min 0.0001 --optimizer SGD"

methods=("ce"
"class_weighted_ce"
"cvar"
"hcvar")
lr=0.01
epochs=2000

perform_exp () {
    dataset=($(make_dataset $1))
    echo $dataset
    for loss_type in ${methods[*]}; do
        if [[ $loss_type == 'ce' || $loss_type == 'class_weighted_ce' ]]; then
            flags="--optimizer SGD"
            run "${loss_type}" ${dataset[0]} ${dataset[1]} ${dataset[2]} $flags --loss_type $loss_type --epochs $epochs --lr $lr --batch_size 1000000
        elif [[ $loss_type == 'hcvar' ]]; then
            alpha=0.01
            for temp in 0.8 1.0 1.2; do
                hcvar_flags="--hcvar_alpha $alpha --hcvar_temp $temp"
                flags="$hcvar_flags $lr_decay_flags"
                run "${loss_type}" ${dataset[0]} ${dataset[1]} ${dataset[2]} $flags --loss_type $loss_type --epochs $epochs --lr $lr --batch_size 1000000
            done
        else
            for alpha in 0.01 0.05 1; do
                cvar_flags="--cvar_alpha $alpha"
                flags="$cvar_flags $lr_decay_flags"
                run "${loss_type}" ${dataset[0]} ${dataset[1]} ${dataset[2]} $flags --loss_type $loss_type --epochs $epochs --lr $lr --batch_size 1000000
            done
        fi
    done
}

perform_exp "covtype"

