export GOOGLE_API_KEY="YOUR_API_KEY"

# Node Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model gemini_api \
    --model_args "model_version=gemini-2.5-flash,response_persistent_folder=YOUR_PATH_TO_RESPONSE_FOLDER" \
    --tasks virbench_node_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH

# Edge Prediction
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model gemini_api \
    --model_args "model_version=gemini-2.5-flash,response_persistent_folder=YOUR_PATH_TO_RESPONSE_FOLDER" \
    --tasks virbench_edge_prediction \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH