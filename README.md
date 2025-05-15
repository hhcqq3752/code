To run inference to generate preference attributes or annotations, run:

```shell
cd inference
MODEL_PATH=meta-llama/Llama-3.3-70B-Instruct
N=1
TEMPERATURE=0.6
TOP_P=0.9
MAX_TOKENS=32768
NUM_WORKERS=1024
DP=4
TP=2
MAX_PASSES=10
./register_server_sglang.sh $MODEL_PATH 32768 $TP
sleep 180

idx=0
INPUT_PATH=SynPref-15M_shard${idx}.jsonl
OUTPUT_PATH=SynPref-15M_shard${idx}_res.jsonl
task="attribute_labeling"  # or "annotation"
python run_attribute_labeling.py \
    --model_name_or_path $MODEL_PATH \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --n $N \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens $MAX_TOKENS \
    --num_workers $NUM_WORKERS \
    --dp $DP \
    --max_passes $MAX_PASSES \
    --task $task
```

To train the reward model, we first convert the dataset from a jsonl file to huggingface dataset format via

```shell
cd preprocess
python jsonl_to_hf.py \
    --input_path SynPref-15M.jsonl \
    --output_path SynPref-15M.hf
python tokenize_dataset.py \
    --model_name Qwen/Qwen3-1.7B \
    --dataset_path SynPref-15M
```

Then, we train the reward model via

```shell
cd train
model_name=Qwen/Qwen3-1.7B
dataset_name=SynPref-15M

lr=4e-5
lr_scheduler=linear
epochs=2
batch_size=256
seed=1

run_name=${model_name}_${dataset_name}_${lr}_${batch_size}_${lr_scheduler}_${epochs}ep_${seed}
accelerate launch \
    --config_file zero1.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port 29501 \
    --machine_rank $RANK \
    --num_processes $((WORLD_SIZE * 8)) \
    --num_machines $WORLD_SIZE \
    run_reward_modeling.py \
    --train_set_path ${dataset_name} \
    --max_length 16384 \
    --model_name ${model_name} \
    --learning_rate ${lr} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type ${lr_scheduler} \
    --warmup_ratio 0.03 \
    --run_name $(basename $run_name) \
    --output_dir $run_name \
    --save_steps 100 \
    --save_total_limit 999999 \
    --seed ${seed} \
    --resume_from_checkpoint
```
