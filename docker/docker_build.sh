#!/bin/bash
IMAGE_NAME="open_duck"
 
docker build --network=host -t $IMAGE_NAME -f Dockerfile .. 
