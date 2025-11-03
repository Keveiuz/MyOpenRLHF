export CUDA_VISIBLE_DEVICES=0,1,2,3
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset experiment-4/train/guardreasoner-100K.parquet \
   --input_key question \
   --output_key response \
   --apply_chat_template \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --max_samples 100000 \
   --pretrain /public/home/ljt/hf_models/Qwen3-8B \
   --save_path experiment-4/model/guardreasoner-sft-100K \
   --ckpt_path experiment-4/model/guardreasoner-sft-100K \
   --save_steps 98 \
   --max_ckpt_num 5 \
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
    # --adam_offload \
    # --load_checkpoint \
    # --wandb [WANDB_TOKENS]
    # --packing_samples
    # --flash_attn

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
