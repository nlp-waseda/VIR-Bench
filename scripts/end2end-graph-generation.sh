export GOOGLE_API_KEY="YOUR_API_KEY"

# End-to-End Graph Generation (video only -> full visiting order graph)
# Add num_concurrent=N to model_args to process multiple videos in parallel.
python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model gemini_api \
    --model_args "model_version=gemini-2.5-flash,response_persistent_folder=YOUR_PATH_TO_RESPONSE_FOLDER" \
    --tasks virbench_end2end_graph_generation \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH

python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model gemini_api \
    --model_args "model_version=gemini-3.7-flash,response_persistent_folder=YOUR_PATH_TO_RESPONSE_FOLDER" \
    --tasks virbench_end2end_graph_generation \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH
