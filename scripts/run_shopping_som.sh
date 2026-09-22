#!/bin/bash
### This script runs the GPT-4V + SoM models on the entire VWA shopping test set.

model="gpt-5.6-luna"
domain="shopping"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
observation=image_som

#viewport
width=1024
height=2048

# Define the batch size variable
batch_size=50

# Define the starting and ending indices
start_idx=0
end_idx=$((start_idx + batch_size))
max_idx=100
date=$(date '+%Y-%m-%d %H:%M:%S')

# Loop until the starting index is less than or equal to max_idx.
while [ $start_idx -le $max_idx ]
do
    bash scripts/reset_shopping.sh
    bash prepare.sh
    python run.py \
     --instruction_path $instruction_path \
     --test_start_idx $start_idx \
     --test_end_idx $end_idx \
     --model $model \
     --result_dir="$model-$domain-$width-x-$height-$date" \
     --test_config_base_dir=config_files/vwa/test_shopping \
     --repeating_action_failure_th 5 --viewport_width $width --viewport_height $height --max_obs_length 3840 \
     --action_set_tag som  --observation_type $observation

    # Increment the start and end indices by the batch size
    start_idx=$((start_idx + batch_size))
    end_idx=$((end_idx + batch_size))

    # Ensure the end index does not exceed 466 in the final iteration
    if [ $end_idx -gt $max_idx ]; then
        end_idx=$max_idx
    fi
done
