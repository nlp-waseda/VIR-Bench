# Node Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=256,min_pixels=37632,max_pixels=75264,attn_implementation=flash_attention_2,device_map=auto" \
    --tasks virbench_node_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH

# Edge Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=256,min_pixels=37632,max_pixels=75264,attn_implementation=flash_attention_2,device_map=auto" \
    --tasks virbench_edge_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH