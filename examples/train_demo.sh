#!/bin/bash

NET_TYPE="high"
RESTART_CONFIG="false"
CHECK_INTERVAL=1200
export NCCL_IB_TIMEOUT=24
export NCCL_PROFILE_PRIMS=0
export NCCL_PROFILE_PRIMS_ENABLE=0
export NCCL_NVLS_ENABLE=0
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_TOPO_AFFINITY=0
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=0
fi

export HOST_GPU_NUM=8
# 当前机器ip
# export LOCAL_IP=$LOCAL_IP
# 多节点机器ip，逗号隔开
# export NODE_IP_LIST="${ip1}:8,${ip2}:8"
# 机器节点个数
export NODES=16
export NODE_NUM=$((${NODES} * ${HOST_GPU_NUM}))

export NCCL_DEBUG=WARN

model_path=/OPENSource/Hunyuan-A52B-Instruct-32k-Preview/241016/global_step614
tokenizer_path=../models
train_data_file=data/car_train.jsonl
# train_data_file=example_data.jsonl

# ds_config_file=ds_zero2_no_offload.json
# ds_config_file=ds_zero3_no_offload.json
ds_config_file=ds_zero3_offload_no_auto.json

output_path=output_path

mkdir -p ${output_path}

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${current_time}.txt"

echo $NODE_IP_LIST > env.txt 2>&1 &
sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" >  "pssh.hosts"
export CHIEF_IP=$LOCAL_IP

HOST_PATH=hostfile
# HOST_PATH=none

deepspeed --hostfile=$HOST_PATH --master_addr $CHIEF_IP ../train/train.py \
    --do_train \
    --model_name_or_path ${model_path} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --train_data_file ${train_data_file} \
    --deepspeed ${ds_config_file} \
    --output_dir ${output_path} \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 1 \
    --max_steps 100 \
    --save_steps 40 \
    --learning_rate 5e-6 \
    --min_lr 6e-7 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --save_safetensors False \
    --hidden_size 6400 \
    --intermediate_size 18304 \
    --model_max_length 4096 \
    --max_seq_length 4096 \
    --moe_topk 1 \
    --num_experts 2 \
    --num_attention_heads 80 \
    --num_key_value_heads 8 \
    --num_layers 4 \
    --cla_share_factor 2 \
    --use_cla \
    --use_mixed_mlp_moe \
    --num_shared_expert 1 \
    --use_qk_norm \
    --bf16 | tee ${log_file}
    # --use_lora \
    # --lora_rank 64 \
    # --lora_alpha 128 \
    # --lora_dropout 0.1 \