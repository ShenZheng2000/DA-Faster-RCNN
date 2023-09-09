run_training() {
    # Set your output path based on the parameter
    baseline="$1"
    output_path="./output/${baseline}"
    warp="$2"
    source_trainset="$3"

    # export NCCL_DEBUG=INFO

    nohup \
    python -m torch.distributed.launch \
            --nproc_per_node=3 \
            uda_train.py \
            --output-path "${output_path}" \
            --warp-aug-lzu "${warp}" \
            --source-trainset "${source_trainset}" \
          > "${baseline}.out" 2>&1 &
}

# run_training "baseline_9_8"
run_training "warp_9_8" True "bdd100k_day_train_valid_vp"