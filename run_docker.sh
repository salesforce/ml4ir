#!/bin/bash
set -ex

# Upgrade docker-compose
echo `docker-compose version`

# Get the job type
JOB_TYPE=$1
shift 1


if [[ $JOB_TYPE == "ml4ir" ]]; then
    # If no command is specifed, run tests
    ML4IR_CMD=""
    if [[ ! -z "$@" ]]; then
        export ML4IR_CMD=""$(printf "%q " "$@")
    else
        unset ML4IR_CMD
    fi

    # Build/Pull ml4ir
    if [[ -z $IMAGE_TAG ]] || [[ $IMAGE_TAG == "dev" ]]; then
        docker-compose build ml4ir
    else
        docker-compose pull ml4ir
    fi

    # Run ml4ir
    docker-compose up \
        --abort-on-container-exit \
        ml4ir

    # Unset ml4ir command env variable
    unset ML4IR_CMD
else
    echo "!!! Wrong Job Type !!!"
    echo "Please choose one from {'ml4ir'}"
fi

# Stop containers
docker-compose down