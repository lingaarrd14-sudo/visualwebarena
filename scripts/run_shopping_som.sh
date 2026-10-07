#!/bin/bash
### This script runs the GPT-4V + SoM models on the entire VWA shopping test set.

provider="openai"
model="gpt-6-luna"
mode="chat"
domain="shopping"
# Select the prompt that enables JSON actions.
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s_json_cot.json"
observation="image_som"
action_set_tag="som"

#viewport
width=1024
height=720

# Define the batch size variable
batch_size=50

# Define the starting and ending indices
start_idx=0
end_idx=$((start_idx + batch_size))
max_idx=466

date=$(date '+%Y-%m-%d')

# Run each non-empty half-open task range [start_idx, end_idx).
while [ $start_idx -lt $max_idx ]
do
    bash scripts/reset_shopping.sh
    bash prepare.sh
    # Keep JSON results separate from the backtick runs.
    python run.py \
     --instruction_path $instruction_path \
     --test_start_idx $start_idx \
     --test_end_idx $end_idx \
     --model $model \
     --provider $provider \
     --mode $mode \
     --result_dir=cache/"${model}-${action_set_tag}-json-cot-${domain}-${width}x${height}-${date}" \
     --test_config_base_dir=config_files/vwa/test_shopping \
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
