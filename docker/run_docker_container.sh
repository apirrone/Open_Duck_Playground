#!/bin/bash
IMAGE_NAME="open_duck"

# Allow local connections to X server (optional; adjust as needed)
# xhost +local:root

docker run --gpus all --rm -it --network=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v $(pwd)/../:/open_duck \
    $IMAGE_NAME
