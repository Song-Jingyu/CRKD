#!/usr/bin/env bash
USER_ID=$(id -u)
GROUP_ID=$(id -g)
PASSWD_FILE=$(mktemp) && echo $(getent passwd $USER_ID) > $PASSWD_FILE
GROUP_FILE=$(mktemp) && echo $(getent group $GROUP_ID) > $GROUP_FILE

docker run -it \
    -e HOME \
    -u $USER_ID:$GROUP_ID \
    -v $PASSWD_FILE:/etc/passwd:ro \
    -v $GROUP_FILE:/etc/group:ro \
    -v /mnt/ws-frb/users/lingjunz/jingyu-docker/docker/home:$HOME \ # TODO: change the directory path before run!
    -v /mnt:/mnt \
    --name lingjunz_bevfusion_pelican \ # TODO: change the container name before use!
    --gpus=all \
    --ipc=host \
    bevfusion:latest # TODO: change this image name, container name and home directory accordingly
