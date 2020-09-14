#!/usr/bin/env bash

CMD=$(printf "%q " "$@")

echo "-------------------------"
echo "RUNNING: $CMD"
echo "-------------------------"

eval $CMD

exit 0
