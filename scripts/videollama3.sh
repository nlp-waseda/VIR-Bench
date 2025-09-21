# Node Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model videollama3 \
    --model_args "pretrained=DAMO-NLP-SG/VideoLLaMA3-7B,max_num_frames=180,device_map=auto,use_custom_video_loader=True" \
    --tasks virbench_node_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH

# Edge Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model videollama3 \
    --model_args "pretrained=DAMO-NLP-SG/VideoLLaMA3-7B,max_num_frames=180,device_map=auto,use_custom_video_loader=True" \
    --tasks virbench_edge_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH