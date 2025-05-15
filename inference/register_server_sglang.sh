#!/bin/bash

# Check that enough arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <model_path> <max_model_len> <tp>"
    exit 1
fi

MODEL_PATH=$1
MAX_MODEL_LEN=$2
TP=$3

# Ensure that TP divides 8 evenly
if [ $((8 % TP)) -ne 0 ]; then
    echo "Error: With 8 GPUs available, tp must be a divisor of 8."
    exit 1
fi

# Calculate the number of servers to launch
NUM_SERVERS=$((8 / TP))

for (( i=0; i<NUM_SERVERS; i++ )); do
    # Build the comma-separated list of GPU indices for this server.
    GPUS=""
    for (( j=0; j<TP; j++ )); do
        GPU_INDEX=$(( i + j * NUM_SERVERS ))
        if [ $j -gt 0 ]; then
            GPUS+=",${GPU_INDEX}"
        else
            GPUS+="${GPU_INDEX}"
        fi
    done

    echo "Starting server on port $((8000+i)) with GPUs: $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS python -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --mem-fraction-static 0.9 \
        --tp $TP \
        --host 127.0.0.1 \
        --port $((8000+i)) \
        --context-length $MAX_MODEL_LEN \
        --allow-auto-truncate \
    &
done

echo "All servers are running"
