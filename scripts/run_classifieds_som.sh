#!/bin/bash

## Define the model, result directory, and instruction path variables, observation
model="gpt-5.6-luna"
domain="classifieds"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
captioning_model="Salesforce/blip2-flan-t5-xl"
observation=image_som

#viewport
width=1024
height=2048

# Define the batch size variable
batch_size=50

# Define the starting and ending indices
start_idx=0
end_idx=$((start_idx + batch_size))
max_idx=63
date=$(date '+%Y-%m-%d %H:%M:%S')

# Run each non-empty half-open task range [start_idx, end_idx).
while [ $start_idx -lt $max_idx ]
do
    # Classifieds reset is quick, so we can do it after every example.
    curl -X POST http://127.0.0.1:9980/index.php?page=reset -d "token=4b61655535e7ed388f0d40a93600254c"
    bash prepare.sh
    python run.py \
     --instruction_path $instruction_path \
     --test_start_idx $start_idx \
     --test_end_idx $end_idx \
     --model $model \
     --result_dir="$model-$domain-$width-x-$height-$date" \
     --test_config_base_dir=config_files/vwa/test_classifieds \
     --repeating_action_failure_th 5 --viewport_width $width --viewport_height $height --max_obs_length 3840 \
     --captioning_model $captioning_model \
     --action_set_tag som  --observation_type $observation

    # Increment the start and end indices by the batch size
    start_idx=$((start_idx + batch_size))
    end_idx=$((end_idx + batch_size))

    # Ensure the end index does not exceed 466 in the final iteration
    if [ $end_idx -gt $max_idx ]; then
        end_idx=$max_idx
    fi
done
