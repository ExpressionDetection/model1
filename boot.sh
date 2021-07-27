#!/bin/bash

source ~/.bashrc

POETRY_VIRTUALENVS_CREATE=false poetry install --no-interaction --no-ansi

python3 /app/model1/server.py