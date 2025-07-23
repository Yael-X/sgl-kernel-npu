#! /bin/bash

nproc_per_node=8
master_addr=141.61.105.141
export MASTER_ADDR=$master_addr
port=17621
export LOGLEVEL=INFO
torchrun --nnodes=1 --nproc-per-node=$nproc_per_node --node_rank=0 --master_addr=$master_addr --master_port=$port ./tests/test_low_latency.py
