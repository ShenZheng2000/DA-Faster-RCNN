# export NCCL_DEBUG=INFO

nohup \
python -m torch.distributed.launch \
        --nproc_per_node=3 \
        uda_train.py \
        --output-path "./output/baseline_9_8" \
      > baseline_9_8.out 2>&1 &