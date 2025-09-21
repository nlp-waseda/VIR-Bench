# Node Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava_vid \
    --model_args "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,device_map=auto" \
    --tasks virbench_node_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH

# Edge Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava_vid \
    --model_args "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,device_map=auto" \
    --tasks virbench_edge_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH