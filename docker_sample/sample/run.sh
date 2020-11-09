USER_NAME=crsprtvision

drun() {
  docker run \
    -v /hdd_ext:/home/$USER_NAME/hdd_ext \
    -v ${HOME}/Projects:/home/$USER_NAME/Projects \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=${DISPLAY} \
    --gpus all \
    --net host \
    --shm-size 8G \
    --privileged=true \
    --name ${USER_NAME} \
    -it \
    --rm \
    -d \
    $1 \
    bash
}

dexec() {
  docker exec $1 bash -c "xauth add $(xauth list | grep unix:$(echo $DISPLAY | cut -d. -f1 | cut -d: -f2))" && docker exec -it $1 bash
}
