# make sure to have the lastest version of transformers bitsandbytes accelerate peft flash-attention

# training setup
NUM_NODES=1
GPUS_PER_NODE=8 # you can go down to 4 gpu
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# go to the trl folder
cd trl 

export CMD=" \
    ./examples/scripts/sft.py \
    --model_name_or_path="meta-llama/Meta-Llama-3.1-405B" \
    --dataset_name="HuggingFaceH4/no_robots" \
    --report_to="wandb" \
    --bf16 true \
    --learning_rate=1e-05 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --output_dir="big-boi-llama3.1" \
    --logging_steps=1 \
    --eval_strategy="no" \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --attn_implementation="eager" \
    --torch_dtype="bfloat16" \
    --bnb_4bit_quant_type="nf4" \
    --use_peft true \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    "
export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 accelerate launch \
    --config_file ./examples/accelerate_configs/fspd_qlora.yaml  \
    --gradient_accumulation_steps 1 \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --role \$(hostname -s): \
    --tee 3 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID -u bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"
