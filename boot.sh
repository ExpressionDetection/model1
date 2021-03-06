#!/bin/bash

source ~/.bashrc

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
        inotifywait -e close_write -r /app/model1/server.py
        kill $PID
    done
  else
    python3 /app/model1/server.py
fi