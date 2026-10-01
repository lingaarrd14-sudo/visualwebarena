#!/bin/bash

### Define the model, result directory, and instruction path variables, observation
provider="openai"
model="gpt-5.6-luna"
mode="chat"
domain="reddit"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
observation="image_som"
action_set_tag="som"

#viewport
width=1024
height=720

# Define the batch size variable
batch_size=30

# Define the starting and ending indices
start_idx=0
end_idx=$((start_idx + batch_size))
max_idx=50

date=$(date '+%Y-%m-%d')

# Run each non-empty half-open task range [start_idx, end_idx).
while [ $start_idx -lt $max_idx ]
do
    # Run the scripts and the Python command with the current indices and defined variables
    bash scripts/reset_reddit.sh
    bash prepare.sh
    python run.py \
     --instruction_path $instruction_path \
     --test_start_idx $start_idx \
     --test_end_idx $end_idx \
     --model $model \
     --provider $provider \
     --mode $mode \
     --result_dir=cache/"${model}-${domain}-${width}x${height}-${date}" \
     --test_config_base_dir=config_files/vwa/test_reddit \
     --repeating_action_failure_th 5 --viewport_width $width --viewport_height $height \
     --action_set_tag $action_set_tag  --observation_type $observation

    # Increment the start and end indices by the batch size
    start_idx=$((start_idx + batch_size))
    end_idx=$((end_idx + batch_size))

    # Ensure the end index does not exceed 466 in the final iteration
    if [ $end_idx -gt $max_idx ]; then
        end_idx=$max_idx
    fi
done
