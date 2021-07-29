#!/bin/bash

source ~/.bashrc

# Install dependencies during container boot
poetry install --no-interaction --no-ansi

if [ $RELOAD_APP_ON_FILE_CHANGE == "true" ]
  then

    sigint_handler()
    {
      kill $PID
      exit
    }

    trap sigint_handler SIGINT

    # Reload server whenever a file is saved
    while true; do
        python3 /app/model1/server.py &
        PID=$!
        inotifywait -e close_write -r `pwd`
        kill $PID
    done
fi