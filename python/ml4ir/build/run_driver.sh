#!/usr/bin/env bash

# TODO: Define default parameters.sh file
CMD=$(printf "%q " "$@")

echo "-------------------------"
echo "RUNNING: $CMD"
echo "-------------------------"

eval $CMD

exit 0
