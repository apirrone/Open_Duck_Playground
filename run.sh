#!/bin/bash
set -e

# Run Open Duck Mini V2 with Python 3.12
# Any arguments passed to this script will be forwarded to the runner

# Set number of timesteps (in millions)
NUM_TIMESTEPS=300000000  # 300M steps

echo "Running Open Duck Mini V2 with Python 3.12..."
echo "Training for $((NUM_TIMESTEPS/1000000))M steps"
uv run -p 3.12 playground/open_duck_mini_v2/runner.py \
  --num_timesteps $NUM_TIMESTEPS \
  --export_onnx \
  --onnx_export_frequency 5 \
  --checkpoint_frequency 2 \
  --eval_step_interval 1 \
  "$@"
