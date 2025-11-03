export CUDA_VISIBLE_DEVICES=0,1,2,3
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset /public/home/ljt/zez/Boundary-Sample/experiment-4/train/guardreasoner-100K-in_distribution-csft-4500.parquet \
   --input_key question \
   --output_key response \
   --apply_chat_template \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --max_samples 50000 \
   --pretrain /public/home/ljt/zez/Boundary-Sample/experiment-4/model/guardreasoner-sft-100K \
   --save_path /public/home/ljt/zez/Boundary-Sample/experiment-4/model/guardreasoner-100K-in_distribution-csft \
   --ckpt_path /public/home/ljt/zez/Boundary-Sample/experiment-4/model/guardreasoner-100K-in_distribution-csft \
   --save_steps -1 \
   --max_ckpt_num 3 \
   --save_hf_ckpt \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --ds_tensor_parallel_size 1 \
   --adam_offload \

EOF
    # --load_checkpoint \
    # --wandb [WANDB_TOKENS]
    # --packing_samples
    # --flash_attn

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
