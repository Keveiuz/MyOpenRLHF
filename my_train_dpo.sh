export CUDA_VISIBLE_DEVICES=0,1,2,3
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --dataset experiment-4/train/guardreasoner-100K-in_distribution-dpo-4500.parquet \
   --save_path experiment-4/model/guardreasoner-100K-in_distribution-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain experiment-4/model/guardreasoner-sft-100K \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --ds_tensor_parallel_size 2 \
   --adam_offload
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
