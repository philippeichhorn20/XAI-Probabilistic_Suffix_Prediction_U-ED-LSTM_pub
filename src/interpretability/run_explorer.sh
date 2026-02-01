#!/bin/bash
# Launch the Attribution Explorer web interface

# Change to script directory
cd "$(dirname "$0")"

# Check if gradio is installed
python -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Gradio not installed. Installing..."
    pip install gradio
fi

# Default paths (adjust as needed)
MODEL_PATH="../notebooks/training_variational_dropout/Helpdesk/Helpdesk_full_grad_norm_philipp_4layer.pkl"
DATASET_PATH="../../encoded_data/test_philipp/helpdesk_all_5_test.pkl"

echo "Starting Attribution Explorer..."
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo ""

# Run with pre-loaded model and dataset
python attribution_explorer.py --model "$MODEL_PATH" --dataset "$DATASET_PATH" "$@"
