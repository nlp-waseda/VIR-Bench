# Node Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model internvl2 \
    --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frames=64,device_map=auto" \
    --tasks virbench_node_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH

# Edge Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model internvl2 \
    --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frames=64,device_map=auto" \
    --tasks virbench_edge_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH